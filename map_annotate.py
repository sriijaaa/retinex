import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = "."   # Current folder
IMG_DIR = os.path.join(BASE_DIR, "originalImages")
EXU_XML_DIR = os.path.join(BASE_DIR, "exudatesLabels")
ODF_XML_DIR = os.path.join(BASE_DIR, "odFoveaLabels")
OUT_DIR = os.path.join(BASE_DIR, "annotated")

# ---------------------------
# Helper: parse Pascal VOC XML
# ---------------------------
def parse_xml(xml_path):
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Cannot parse {xml_path}: {e}")
        return boxes

    for obj in root.findall("object"):
        label = obj.find("name").text.strip() if obj.find("name") is not None else "object"
        bnd = obj.find("bndbox")
        if bnd is not None:
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))
            boxes.append({"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return boxes

# ---------------------------
# Draw legend on image
# ---------------------------
def draw_legend(img):
    legend_items = [
        ("Exudates", (0, 255, 255)),  # Yellow
        ("Fovea", (255, 0, 0)),       # Blue
        ("Optic Disc", (0, 0, 255)),  # Red
        ("Other", (0, 255, 0))        # Green
    ]
    x, y0 = 10, 30
    for i, (name, color) in enumerate(legend_items):
        y = y0 + i * 30
        cv2.rectangle(img, (x, y - 20), (x + 20, y), color, -1)
        cv2.putText(img, name, (x + 30, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# ---------------------------
# Main function for one image
# ---------------------------
def annotate_one_image(image_filename, show=True):
    # Paths
    img_path = os.path.join(IMG_DIR, image_filename)
    exu_xml_path = os.path.join(EXU_XML_DIR, os.path.splitext(image_filename)[0] + ".xml")
    odf_xml_path = os.path.join(ODF_XML_DIR, os.path.splitext(image_filename)[0] + ".xml")

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not open {img_path}")
        return

    # Parse XML files if exist
    boxes = []
    if os.path.exists(exu_xml_path):
        boxes.extend(parse_xml(exu_xml_path))
    if os.path.exists(odf_xml_path):
        boxes.extend(parse_xml(odf_xml_path))

    # Show all unique labels for debugging
    unique_labels = set([box["label"] for box in boxes])
    print(f"[INFO] Unique labels in {image_filename}: {unique_labels}")

    print(f"[INFO] Found {len(boxes)} boxes for {image_filename}")

    # Draw boxes
    for box in boxes:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        label = box["label"].lower()

        # Assign colors based on label variations
        if "exudate" in label or label.startswith("ex"):
            color = (0, 255, 255)   # Yellow
        elif "fovea" in label:
            color = (255, 0, 0)     # Blue
        elif "od" in label or "disc" in label:
            color = (0, 0, 255)     # Red
        else:
            color = (0, 255, 0)     # Green (default/other)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, box["label"], (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Add legend
    draw_legend(img)

    # Save annotated output
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, image_filename)
    cv2.imwrite(out_path, img)

    if show:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    print(f"[SAVED] {out_path}")

# ---------------------------
# Example run (test with one image)
# ---------------------------
if __name__ == "__main__":
    # Change filename here for testing
    annotate_one_image("1209.jpg", show=True)
