import os
import shutil

SOURCE_DIR = "dataset"
TRAIN_DIR = "train"
TEST_DIR = "test"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_images(src, dest):
    print(f"Copying from: {src} to: {dest}")
    for root, dirs, files in os.walk(src):
        print(files)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest, file)
                print('srcpath',src_path,'dest',dest_path)
                print(f" - Copying {src_path} → {dest_path}") # debug print
                shutil.copy(os.path.join(root, file), dest)
            else:
                print('if not wworked')

def process_person(person_name):
    print(f"Processing: {person_name}")
    person_path = os.path.join(SOURCE_DIR, person_name)

    # Create train and test folders for the person
    train_person_path = os.path.join(TRAIN_DIR, person_name)
    test_person_path = os.path.join(TEST_DIR, person_name)
    ensure_dir(train_person_path)
    ensure_dir(test_person_path)

    # Copy lowquality images to train
    lowq_path = os.path.join(person_path, "low_quality")
    print(lowq_path,'lowq_path')
    copy_images(lowq_path, train_person_path)

    # Copy highquality images to test
    highq_path = os.path.join(person_path, "high_quality")
    print(highq_path,'highq_path')
    copy_images(highq_path, test_person_path)

def main():
    for person in os.listdir(SOURCE_DIR):
        person_path = os.path.join(SOURCE_DIR, person)
        print('person path:',person_path)
        if os.path.isdir(person_path):
            process_person(person)

if __name__ == "__main__":
    main()