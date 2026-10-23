import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_engineering import FeatureEngineeringPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo feature baseline cho sentiment và aspect")
    parser.add_argument("--input", default="data/processed/reviews_processed.csv")
    parser.add_argument("--output-dir", default="data/processed/features")
    parser.add_argument("--aspects", default="data/dictionaries/aspect_keywords.json")
    args = parser.parse_args()

    pipeline = FeatureEngineeringPipeline(aspect_keywords_path=args.aspects)
    metadata = pipeline.run(input_path=args.input, output_dir=args.output_dir)
    print("=" * 64)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 64)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()