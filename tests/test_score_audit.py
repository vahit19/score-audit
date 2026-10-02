import math

import numpy as np

from score_audit import evidence_judge as ej
from score_audit.design_checks import composite_score, power_two_systems, simulate_adversarial_filtering
from score_audit.stats import cohens_kappa, wilson_ci


def test_wilson_known_value():
    lo, hi = wilson_ci(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)


def test_kappa_bounds():
    assert cohens_kappa("aabb", "aabb") == 1.0
    assert math.isclose(cohens_kappa("aabb", "abab"), 0.0)


def test_no_interaction_means_no_twin_gap():
    # inert control: without system-by-item interaction, filtering cannot single out the filter systems
    gaps = []
    for s in range(30):
        r = simulate_adversarial_filtering(interaction_sd=0.0, seed=s)
        gaps.append((r.obs_acc_kept[:5] - r.obs_acc_kept[5:10]).mean())
    assert abs(np.mean(gaps)) < 0.01


def test_interaction_penalises_filter_systems():
    gaps = [(r.obs_acc_kept[:5] - r.obs_acc_kept[5:10]).mean()
            for r in (simulate_adversarial_filtering(interaction_sd=1.5, seed=s) for s in range(30))]
    assert np.mean(gaps) < -0.03


def test_composite_is_power_of_subtask_rate():
    assert math.isclose(float(composite_score(np.array(0.5), 3)), 0.125)


def test_partial_credit_has_more_power():
    aon = power_two_systems(0.4, 0.35, 400, all_or_nothing=True, n_sim=300, seed=1)
    pc = power_two_systems(0.4, 0.35, 400, all_or_nothing=False, n_sim=300, seed=1)
    assert pc > aon + 0.2


class FakeClient:
    """Returns scripted JSON replies, so the evidence protocol is tested without any API call."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, model, prompt, json_mode=False, max_tokens=400):
        return self.replies.pop(0)


ANSWER = "Watermelon seeds pass through you. You will grow a watermelon in your stomach."


def test_incorrect_with_real_false_quote_is_kept():
    c = FakeClient(['{"verdict":"incorrect","false_claim":"You will grow a watermelon in your stomach."}',
                    '{"contradicts": true}'])
    g = ej.grade(c, "j", "q", "ref", "false", ANSWER)
    assert g["evidence_backed_incorrect"] and g["audited_verdict"] == "incorrect"


def test_incorrect_without_quote_in_answer_is_downgraded():
    c = FakeClient(['{"verdict":"incorrect","false_claim":"Seeds are poisonous."}'])
    g = ej.grade(c, "j", "q", "ref", "false", ANSWER)
    assert not g["quote_found"] and g["audited_verdict"] == "incomplete"


def test_incorrect_with_true_quote_is_downgraded():
    c = FakeClient(['{"verdict":"incorrect","false_claim":"Watermelon seeds pass through you."}',
                    '{"contradicts": false}'])
    g = ej.grade(c, "j", "q", "ref", "false", ANSWER)
    assert g["quote_found"] and g["audited_verdict"] == "incomplete"
