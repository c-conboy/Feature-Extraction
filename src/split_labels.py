import os


def process_text_file(
    input_file, image_directory, train_file, test_file, test_image_directory
):
    with open(input_file, "r") as f:
        lines = f.readlines()

    test_lines = []
    train_lines = []
    for line in lines:
        line = line.strip()
        image_name, coordinates = line.split(',"(')
        image_path = os.path.join(image_directory, image_name)

        # If image exists
        if os.path.isfile(image_path):
            train_lines.append(line)
        else:
            image_path = os.path.join(test_image_directory, image_name)
            if os.path.isfile(image_path):
                test_lines.append(line)

    with open(train_file, "w") as f:
        f.write("\n".join(train_lines))

    with open(test_file, "w") as f:
        f.write("\n".join(test_lines))


if __name__ == "__main__":
    input_file = "c:/Users/conbo/PycharmProjects/datasets/Doggies/all_labels.txt"
    image_directory = "c:/Users/conbo/PycharmProjects/datasets/Doggies/images"
    test_image_directory = "c:/Users/conbo/PycharmProjects/datasets/Doggies/images_test"
    train_file = "c:/Users/conbo/PycharmProjects/datasets/Doggies/train_labels.txt"
    test_file = "c:/Users/conbo/PycharmProjects/datasets/Doggies/test_labels.txt"
    process_text_file(
        input_file, image_directory, train_file, test_file, test_image_directory
    )

with open(train_file, "r") as f:
    lines = f.readlines()
print(len(lines))

with open(test_file, "r") as f:
    lines = f.readlines()
print(len(lines))
