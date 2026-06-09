# see https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-utils
import logging
import time
from typing import Any, Dict, List

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "math": (
        "Please provide only the final result (a number or LaTeX formula) within the <answer>...</answer> tags.\n"
        "Example:\n<answer>$$-\\dfrac{3}{2}$$</answer>"
    ),
    "spatial grounding": (
        "Please provide only the bounding box as JSON with key 'boxes' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"boxes\": [35, 227, 437, 932]}</answer>"
    ),
    "trace": (
        "Please provide only the ordered waypoints as JSON with key 'point_2d' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>```json\n[{\"point_2d\": [624, 469]}, {\"point_2d\": [640, 421]}, {\"point_2d\": [638, 372]}, {\"point_2d\": [613, 337]}]\n```</answer>"
    ),
    "trace_3d": (
        "Please provide only the ordered 2D waypoints with depth (in meters) as JSON with key 'point_2d' and 'depth' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>```json\n[{\"point_2d\": [463, 599], \"depth\": 1.08}, {\"point_2d\": [458, 603], \"depth\": 1.08}, {\"point_2d\": [449, 612], \"depth\": 1.06}]\n```</answer>"
    ),
    "point": (
        "Please pointing to answer the question within the <answer>...</answer> tags.\n"
        "Example:\n<answer>```json\n[{\"point_2d\": [230, 138]}]\n```</answer>"
    ),
}


class HuggingFaceClient:
    """Client for HuggingFace model inference with support for images and videos"""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "auto",
    ):
        """Initialize the HuggingFace client with model and processor"""
        logger.info(f"Loading model from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map
        )
        logger.info("Model loaded successfully")

    def prepare_messages(
        self,
        test_case: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build messages from test case data"""
        content = []
        prompt_text = QUESTION_TEMPLATE.format(Question=test_case["prompt"])
        test_type = test_case["type"]

        if test_type == "single_image":
            # Local single image with file:// prefix
            image_path = test_case["image"]
            content.append({"type": "image", "image": f"file://{image_path}"})
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "multi_image":
            # Local multiple images with file:// prefix
            for img_path in test_case["images"]:
                content.append({"type": "image", "image": f"file://{img_path}"})
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "single_pil_image":
            # Single PIL image (expects PIL Image object in test_case)
            pil_image = test_case["image"]
            content.append({"type": "image", "image": pil_image})
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "multi_pil_image":
            # Multiple PIL images (expects list of PIL Image objects in test_case)
            for pil_image in test_case["images"]:
                content.append({"type": "image", "image": pil_image})
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "video":
            # Local video with file:// prefix
            video_path = test_case["video"]
            content.append({"type": "video", "video": f"file://{video_path}"})
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "video_frames":
            sample_fps = test_case.get("sample_fps", 1)
            # Local video frames with file:// prefix
            frame_paths = [f"file://{frame}" for frame in test_case["video_frames"]]
            content.append({"type": "video", "video": frame_paths, "sample_fps": sample_fps})
            content.append({"type": "text", "text": prompt_text})

        return [{"role": "user", "content": content}]

    def inference(
        self,
        test_case: Dict[str, Any],
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Process a single test case and return results"""
        test_type = test_case["type"]

        try:
            messages = self.prepare_messages(test_case)

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision info
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )

            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            # Prepare inputs
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs
            )
            inputs = inputs.to(self.model.device)

            # Generate
            start_time = time.time()
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            latency = time.time() - start_time

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return {
                "success": True,
                "test_type": test_type,
                "prompt": test_case["prompt"],
                "generated_text": output_text[0],
                "latency": latency,
            }

        except Exception as e:
            logger.error(f"Case failed: {str(e)}")
            return {
                "success": False,
                "test_type": test_type,
                "error": str(e),
            }


if __name__ == "__main__":
    # Model configuration
    model_path = "IffYuan/Embodied-R1.5"

    # Initialize client
    client = HuggingFaceClient(
        model_path=model_path,
        device_map="auto",
        dtype="auto"
    )

    # Test Case 0: Local single image
    print('\n' + '=' * 80)
    print("Case 0 - Type: Local single image")
    print('=' * 80)

    case_0 = {
        "answer": "(C)",
        "prompt": "How many table lamps are in the image? Select from the following choices.\n(A) 0\n(B) 2\n(C) 1\n(D) 3",
        "image": "test_assets/sample_2_image.png",
        "type": "single_image"
    }

    result_0 = client.inference(case_0, max_new_tokens=512)
    print(f"\nPrompt: {result_0['prompt']}")
    print(f"\nGenerated Answer: {result_0['generated_text']}")
    print(f"Latency: {result_0['latency']:.2f}s")
    print('-' * 80)

    # Test Case 1: Local multiple images
    print('\n' + '=' * 80)
    print("Case 1 - Type: Local multiple images (本地多图)")
    print('=' * 80)

    case_1 = {
        "answer": "place sticky notes in the stand",
        "prompt": "current goal is: please stock caddy for phone room. last 20 steps: 1- place black pen in pen stand 2- place black pen in pen stand 3- place black pen in pen stand 4- place black pen in pen stand 5- place blue pen in pen stand 6- place blue pen in pen stand 7- place blue pen in pen stand. what is the immediate next step?",
        "images": [
            "test_assets/robovqa_086678/01.jpg",
            "test_assets/robovqa_086678/02.jpg",
            "test_assets/robovqa_086678/03.jpg",
            "test_assets/robovqa_086678/04.jpg",
            "test_assets/robovqa_086678/05.jpg",
            "test_assets/robovqa_086678/06.jpg",
            "test_assets/robovqa_086678/07.jpg",
            "test_assets/robovqa_086678/08.jpg"
        ],
        "type": "multi_image"
    }

    result_1 = client.inference(case_1, max_new_tokens=512)
    print(f"\nPrompt: {result_1['prompt']}")
    print(f"\nGenerated Answer: {result_1['generated_text']}")
    print(f"Latency: {result_1['latency']:.2f}s")
    print('-' * 80)

    # Test Case 2: Single PIL image
    print('\n' + '=' * 80)
    print("Case 2 - Type: Single PIL image")
    print('=' * 80)

    # Load PIL image first
    pil_image_2 = Image.open("test_assets/sample_0_image.png")

    case_2 = {
        "answer": "door",
        "prompt": "Among the listed objects, which one is closest to your current location in the image? The option: table, towel, door or basket. Direct answer the question",
        "image": pil_image_2,
        "type": "single_pil_image"
    }

    result_2 = client.inference(case_2, max_new_tokens=512)
    print(f"\nPrompt: {result_2['prompt']}")
    print(f"\nGenerated Answer: {result_2['generated_text']}")
    print(f"Latency: {result_2['latency']:.2f}s")
    print('-' * 80)

    # Test Case 3: Multiple PIL images
    print('\n' + '=' * 80)
    print("Case 3 - Type: Multiple PIL images")
    print('=' * 80)

    # Load PIL images first
    pil_images_3 = [
        Image.open("test_assets/aff.png"),
        Image.open("test_assets/sample_0_image.png")
    ]

    case_3 = {
        "answer": "",
        "prompt": "Provide one or more points coordinate of objects region this sentence describes: you need to grasp the mug. The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}].",
        "images": pil_images_3,
        "type": "multi_pil_image"
    }

    result_3 = client.inference(case_3, max_new_tokens=512)
    print(f"\nPrompt: {result_3['prompt']}")
    print(f"\nGenerated Answer: {result_3['generated_text']}")
    print(f"Latency: {result_3['latency']:.2f}s")
    print('-' * 80)

    # # Test Case 4: Local video
    # print('\n' + '=' * 80)
    # print("Case 4 - Type: Local video")
    # print('=' * 80)

    # case_4 = {
    #     "answer": "",
    #     "prompt": "\nThese are frames of a video.\nIf I am standing by the trash can and facing the counter top, is the dog bed to the left or the right of the counter top?\nA. left\nB. right\nAnswer with the option's letter from the given choices directly.",
    #     "video": "test_assets/5a83c3ab-9a60-4c9e-93be-4735d522f3f1.mp4",
    #     "type": "video"
    # }

    # result_4 = client.inference(case_4, max_new_tokens=512)
    # print(f"\nPrompt: {result_4['prompt']}")
    # print(f"\nGenerated Answer: {result_4['generated_text']}")
    # print(f"Latency: {result_4['latency']:.2f}s")
    # print('-' * 80)

    # Test Case 5: Local video frames
    print('\n' + '=' * 80)
    print("Case 5 - Type: Local video frames")
    print('=' * 80)

    case_5 = {
        "answer": "",
        "prompt": "Describe what is happening in these video frames.",
        "video_frames": [
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_000.jpg",
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_001.jpg",
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_002.jpg",
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_003.jpg",
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_004.jpg",
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_005.jpg",
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_006.jpg",
            "test_assets/45b6f83f-0100-4499-a220-d3fc41465d98_frame_007.jpg"
        ],
        'sample_fps': 1,
        "type": "video_frames"
    }

    result_5 = client.inference(case_5, max_new_tokens=512)
    print(f"\nPrompt: {result_5['prompt']}")
    print(f"\nGenerated Answer: {result_5['generated_text']}")
    print(f"Latency: {result_5['latency']:.2f}s")
    print('-' * 80)
