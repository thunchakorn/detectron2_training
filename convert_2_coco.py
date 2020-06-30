from labelme2coco.labelme2coco import labelme2coco
import argparse
import glob
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "labelme_images",
        help="Directory to labelme images and annotation json files.",
        type=str,
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="trainval.json"
    )
    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    labelme2coco(labelme_json, args.output)