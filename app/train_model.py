"""
Lastly, train the model
"""
from openai import OpenAI
from dotenv import load_dotenv
import os, time

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Path to the fine-tuning dataset
training_file_path = "chat_fine_tuning_dataset.jsonl"

def upload_training_file():
    print("Uploading training file...")
    # Upload the training file
    with open(training_file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose="fine-tune"
        )
        print(f"upload_training_file response {response}")
    
    # Access the file ID using attribute notation
    file_id = response.id
    print(f"File uploaded successfully. File ID: {file_id}")
    return file_id

def create_fine_tuning_job(file_id):
    print("Creating fine-tuning job...")
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18"
    )
    print(f"create_fine_tuning_job response {response}")

    fine_tune_id = response.id
    print(f"Fine-tuning job created successfully. Fine-tune ID: {fine_tune_id}")
    return fine_tune_id


def monitor_fine_tuning_job(fine_tune_id):
    print("Monitoring fine-tuning job...")
    status = None
    while status not in ["succeeded", "failed"]:
        # Retrieve the current state of the job
        response = client.fine_tuning.jobs.retrieve(fine_tune_id)
        print(f"monitor_fine_tuning_job response {response}")
        status = response.status
        print(f"Status: {status}")

        if status == "succeeded":
            print(f"Fine-tuning completed successfully. Fine-tuned model: {response.fine_tuned_model}")
            return response.fine_tuned_model
        elif status == "failed":
            print("Fine-tuning job failed.")
            break
        else:
            # Check again after 10 seconds
            time.sleep(10)


def main():
    # Step 1: Upload the training dataset
    file_id = upload_training_file()

    # Step 2: Create the fine-tuning job
    fine_tune_id = create_fine_tuning_job(file_id)

    # Step 3: Monitor the fine-tuning job
    monitor_fine_tuning_job(fine_tune_id)


if __name__ == "__main__":
    main()