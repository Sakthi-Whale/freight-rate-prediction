from src.train import train_pipeline
from src.predict import generate_predictions
from src.utils import print_header

def main():

    print_header(
        "Freight Rate Prediction"
    )

    train_pipeline()

    print("\nGenerating predictions...\n")

    generate_predictions()

if __name__ == "__main__":
    main()