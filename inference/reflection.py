import time
from pathlib import Path

from point_utils import omni_decode_points, vis_point
from vllm_online_example import VLLMOnlineClient


def inference(client, case, vis=False):
    messages = client.prepare_messages(case)
    start_time = time.time()

    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    latency = time.time() - start_time
    generated_text = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens = usage.total_tokens if usage else 0

    print(f"\nPrompt: {case['prompt']}")
    print(f"\nGenerated Answer: {generated_text}")
    print(f"Latency: {latency:.2f}s | Tokens: {prompt_tokens}+{completion_tokens}={tokens}")
    print('-' * 80)

    if vis:
        points = omni_decode_points(generated_text)
        print(f"Decoded points: {points}")

        image_path = case.get("image")
        if image_path:
            vis_dir = Path("vis_outputs")
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_path = vis_dir / f"case_{case.get('idx', 'unknown')}_points.jpg"
            vis_point(image_path, points, str(vis_path))
            print(f"Saved point visualization to: {vis_path}")
        else:
            print("No image found in case, skip visualization.")

    return generated_text


def plan_with_reflection(client, case, max_rounds=3):
    base_prompt = case["prompt"]
    feedback_text = ""

    # Keep all non-prompt inputs fixed across rounds (image/video/type/idx, etc.)
    fixed_case = {k: v for k, v in case.items() if k != "prompt"}

    for round_idx in range(1, max_rounds + 1):
        print(f"\n{'=' * 80}")
        print(f"Planning Round {round_idx}")
        print(f"{'=' * 80}")

        planning_prompt = base_prompt
        if feedback_text:
            planning_prompt = (
                f"{base_prompt}\n\n"
                f"Previous plan was judged as incorrect by reflection.\n"
                f"Full reflection feedback:\n{feedback_text}\n\n"
                f"Please generate a new planning result that addresses this feedback."
            )

        planning_case = dict(fixed_case)
        planning_case["prompt"] = planning_prompt
        plan_text = inference(client, planning_case)

        reflection_prompt = f"""
            This is the beginning stage of a robot executing a task.\nThe robot in the image was given the instruction.

            Instruction and context:
            {planning_prompt}

            Proposed plan:
            {plan_text}

            Is this plan correct?
            If correct, reply with exactly: correct
            If incorrect, What is wrong with the subtask planning?
        """

        reflection_case = dict(fixed_case)
        reflection_case["prompt"] = reflection_prompt

        print(f"\n{'=' * 80}")
        print(f"Reflection Round {round_idx}")
        print(f"{'=' * 80}")

        reflection_text = inference(client, reflection_case)
        if "correct" in reflection_text.lower() and "incorrect" not in reflection_text.lower():
            print("Reflection verdict: plan is correct. Stop replanning loop.")
            return plan_text

        feedback_text = reflection_text
        print(f"Reflection verdict: plan is incorrect. Full feedback:\n{feedback_text}")

    print("Reached maximum reflection rounds. Return last planning result.")
    return plan_text


if __name__ == "__main__":
    # Model and server configuration
    base_url = "http://localhost:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5"

    # Sampling parameters
    seed = 42
    top_p = 0.8
    temperature = 1.0
    presence_penalty = 1.5
    max_tokens = 4096  # out_seq_length

    # Video processing parameters (for mm_processor_kwargs)
    video_fps = 2  # Frames per second for video sampling
    video_do_sample_frames = True  # Enable frame sampling

    # Initialize client
    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        timeout=3600
    )

    print('\n' + '=' * 80)
    print("Case: Cup Planning")
    print('=' * 80)

    # # full planning
    planning_pick_prompt = """
    current goal is: Pick the orange cup.

    What is the next 4 planning steps?

    Rules:
    - A cup can only be picked if no other cup is placed on top of it.
    - A cup that is partially or fully occluded by another cup is NOT pickable.
    - If the goal cup is not immediately pickable, you must first pick the cups that block it.
    - The order of pick actions is the actual execution order.

    current goal is: Pick the orange cup.

    What is the next 4 planning steps?

    """

    case_0 = {
        "idx": 0,
        "prompt": planning_pick_prompt,
        "image": "test_assets/image_raw.png",
        "type": "single_image"
    }

    final_plan = plan_with_reflection(client, case_0, max_rounds=3)
    print(f"\nFinal plan after reflection loop:\n{final_plan}")
