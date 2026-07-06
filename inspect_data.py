import json, re

with open("test.jsonl") as f:
    samples = [json.loads(l) for l in f]

for s in samples[:5]:
    img_name = s["image"].split("/")[-1]
    convs = s["conversations"]
    caption = re.search(r"Caption: (.+?)(?:\n\nCan you)", convs[0]["value"], re.DOTALL).group(1).strip()
    expected_tags = convs[1]["value"]
    expected_correction = convs[3]["value"].replace("Modified caption: ", "").strip()

    print(f"IMAGE: {img_name}")
    print(f"\nPASTE THIS CAPTION:\n{caption}")
    print(f"\nEXPECTED TAGS:\n{expected_tags}")
    print(f"\nEXPECTED CORRECTION:\n{expected_correction}")
    print("\n" + "="*60 + "\n")
