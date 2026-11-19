import json

import pytest

from voguefusion import config
from voguefusion.services import generate_input, load_workflow
from voguefusion.workflow import (
    WorkflowError,
    bind,
    default_negative,
    fill_slots,
    output_node,
    reference_slots,
    substate_for,
)

CFG = config.load(str(config.DEFAULT_CONFIG))


@pytest.mark.parametrize("name", ["v1", "v2"])
def test_bind_sets_values_by_role(name):
    graph, _ = load_workflow(CFG, name)
    g = bind(
        graph, prompt="red paisley", negative="blur", seed=7, batch=2, reference_files=["a.png"]
    )
    assert g["6"]["inputs"]["text"] == "red paisley"
    assert g["7"]["inputs"]["text"] == "blur"
    assert g["3"]["inputs"]["seed"] == 7
    assert g["10"]["inputs"]["amount"] == 2
    assert [g[i]["inputs"]["image"] for i in reference_slots(g)] == ["a.png"] * 4
    assert set(g) == set(graph)  # structure unchanged
    assert graph["6"]["inputs"]["text"] != "red paisley"  # input not mutated


def test_hostile_prompt_text_cannot_change_graph_structure():
    graph, _ = load_workflow(CFG, "v1")
    hostile = '", "clip": ["4", 1]}, "99": {"class_type": "SaveImage", "inputs": {}}, "x": {"'
    g = bind(graph, prompt=hostile, negative="", seed=1, batch=1, reference_files=["r.png"])
    assert g["6"]["inputs"]["text"] == hostile
    assert set(g) == set(graph)
    assert json.loads(json.dumps(g))["6"]["inputs"]["clip"] == ["4", 1]


def test_v2_differs_from_v1_only_by_lora_batch_and_negative():
    v1, _ = load_workflow(CFG, "v1")
    v2, _ = load_workflow(CFG, "v2")
    assert set(v2) - set(v1) == {"40"}
    assert v2["40"]["class_type"] == "LoraLoader"
    assert v2["40"]["inputs"]["lora_name"] == "VogueFusion.safetensors"
    assert v2["20"]["inputs"]["model"] == ["40", 0]
    assert v2["10"]["inputs"]["amount"] == 3
    assert default_negative(v1) == "human"


def test_v1_matches_recovered_sampler_settings():
    v1, _ = load_workflow(CFG, "v1")
    ks = v1["3"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (
        10,
        2.0,
        "lcm",
        "normal",
    )
    assert v1["4"]["inputs"]["ckpt_name"] == "dreamshaperXL_sfwLightningDPMSDE.safetensors"


def test_fill_slots_cycles_and_bounds():
    assert fill_slots(["a", "b"], 4) == ["a", "b", "a", "b"]
    with pytest.raises(WorkflowError):
        fill_slots([], 4)
    with pytest.raises(WorkflowError):
        fill_slots(list("abcde"), 4)


def test_substate_mapping():
    graph, _ = load_workflow(CFG, "v1")
    assert substate_for(graph, "4") == "loading_model"
    assert substate_for(graph, "3") == "sampling"
    assert substate_for(graph, output_node(graph)) == "saving"
    assert substate_for(graph, None) is None


def test_input_key_is_stable_and_sensitive_to_every_field():
    base = dict(workflow="v1", prompt="p", negative=None, seed=1, batch=1, references=["a"])
    _, k1 = generate_input(CFG, **base)
    _, k2 = generate_input(CFG, **base)
    assert k1 == k2
    for field, value in [
        ("prompt", "q"),
        ("seed", 2),
        ("batch", 2),
        ("references", ["b"]),
        ("workflow", "v2"),
        ("negative", "x"),
    ]:
        _, k = generate_input(CFG, **{**base, field: value})
        assert k != k1, field
