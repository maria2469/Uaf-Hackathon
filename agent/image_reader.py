import os
import base64
import json
import time
from typing import Dict, Any
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
IMAGE_DIR = "lab_images"         # Folder containing patient images
OUTPUT_JSON = "image_data.json"  # Output file (auto-updated)
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_RETRIES = 3                  # Retry failed images
SAVE_INTERVAL = 1                # Save after every image

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ================= UTIL FUNCTIONS =================
def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_existing_data() -> Dict[str, Any]:
    """Load existing JSON data if available."""
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Existing JSON corrupted. Starting fresh.")
    return {}


def save_data(all_data: Dict[str, Any]):
    """Save all data to JSON safely."""
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)


# ================= CORE FUNCTION =================
def extract_lab_data_from_images() -> Dict[str, Any]:
    """
    Processes all lab report images and extracts structured data via Groq vision model.
    Saves incrementally after every image.
    """
    all_data = load_existing_data()

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Directory '{IMAGE_DIR}' not found.")
        return {}

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"⚠️ No image files found in '{IMAGE_DIR}'")
        return {}

    print(f"🧠 Processing {len(image_files)} lab report images...\n")

    for img_name in tqdm(image_files, desc="Analyzing Reports"):
        patient_id = os.path.splitext(img_name)[0]
        if patient_id in all_data:
            continue  # Skip already processed

        img_path = os.path.join(IMAGE_DIR, img_name)
        base64_image = encode_image(img_path)

        prompt = """
You are an expert medical lab report analyzer.
Extract EVERY piece of structured data from the attached lab report image, including:

- Patient details (name, age, gender, ID, DOB, contact, address)
- Report metadata (lab name, report date, referred by, technician, doctor)
- All test results (hematology, biochemistry, liver/kidney function, hormones, etc.)
- Reference ranges for each test
- Remarks, doctor comments, or diagnostic impressions

Return data **strictly in a valid JSON object**.
Do NOT include text, explanations, or markdown.

JSON example:
{
  "patient_info": {
    "name": "",
    "patient_id": "",
    "age": "",
    "gender": "",
    "date_of_birth": "",
    "contact": "",
    "address": ""
  },
  "report_details": {
    "report_date": "",
    "report_time": "",
    "referred_by": "",
    "lab_name": "",
    "technician": "",
    "doctor_name": ""
  },
  "test_results": {
    "hematology": {"hemoglobin": 13.2, "WBC": 6500, "RBC": 4.8},
    "biochemistry": {"glucose": 95, "cholesterol": 180}
  },
  "reference_ranges": {
    "hemoglobin": "13.0–17.0 g/dL",
    "glucose": "70–110 mg/dL"
  },
  "remarks": {
    "summary": "",
    "abnormal_findings": "",
    "doctor_comment": ""
  }
}
Omit fields that are not present in the report.
"""

        # ===== Retry logic for robustness =====
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                 },
                            ],
                        }
                    ],
                    temperature=0,
                    max_completion_tokens=2048,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"},
                )

                # Extract content safely
                content = completion.choices[0].message.content
                if isinstance(content, list):
                    json_text = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                else:
                    json_text = str(content).strip()

                structured_data = json.loads(json_text)
                all_data[patient_id] = structured_data

                # Save incrementally
                if SAVE_INTERVAL:
                    save_data(all_data)

                break  # ✅ Success, break retry loop

            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON for {patient_id} (Attempt {attempt}/{MAX_RETRIES})")
                if attempt == MAX_RETRIES:
                    all_data[patient_id] = {"error": "Invalid JSON output"}
                    save_data(all_data)
            except Exception as e:
                print(f"❌ Error processing {patient_id} (Attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES:
                    all_data[patient_id] = {"error": str(e)}
                    save_data(all_data)
                else:
                    time.sleep(2)  # backoff before retry

    print(f"\n✅ Extraction complete. Total processed: {len(all_data)}")
    return all_data


# ================= MAIN =================
if __name__ == "__main__":
    extract_lab_data_from_images()
