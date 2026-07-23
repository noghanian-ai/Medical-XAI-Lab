import os


def create_output_folder(output_root,
                         class_name,
                         image_name):

    image_name = os.path.splitext(image_name)[0]

    output_dir = os.path.join(
        output_root,
        class_name,
        image_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    return output_dir