from PIL import Image
import cv2
import os, random, glob
import numpy as np
import string


def pil_resize(img, size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.resize(size)
    return np.array(img)


def pil_resize_label(label, size):
    resized_img = np.zeros((size[1], size[0], label.shape[-1]), dtype="uint8")
    for i in range(label.shape[-1]):
        resized_img[:, :, i] = pil_resize(label[:, :, i], size)

    return resized_img


def get_masks(label):
    car_mask = label[:, :, 0]
    window_mask = label[:, :, 1]
    window_mask = cv2.cvtColor((window_mask).astype("uint8"), cv2.COLOR_GRAY2BGR)
    car_mask = cv2.cvtColor((car_mask).astype("uint8"), cv2.COLOR_GRAY2BGR)

    return car_mask, window_mask


def Paste_Inbooth_Image_in_Outside(original_image, label, bg_image):
    car_mask, window_mask = get_masks(label)
    carmask_mod = car_mask.copy()
    h, w = original_image.shape[:2]
    bg_image = pil_resize(bg_image, (w, h))
    if random.random() < 0.3:
        kernel = random.choice([11, 21, 31, 41, 51])
        bg_image = cv2.GaussianBlur(bg_image, (kernel, kernel), 0)

    final_image_1 = (original_image * (carmask_mod / 255) + bg_image * (1 - (carmask_mod / 255))).astype("uint8")
    # final_image = call_window_blend(final_image_1, bg_image, window_mask)
    return final_image_1.astype('uint8')


def change_background(original_image, car_mask, window_mask, outside_bgs_path):
    bg_image = cv2.imread(random.choice(outside_bgs_path))
    carmask_mod = car_mask.copy()

    h, w = original_image.shape[:2]
    bg_image = pil_resize(bg_image, (w, h))

    final_image_1 = (original_image * (carmask_mod / 255) + bg_image * (1 - (carmask_mod / 255))).astype("uint8")
    final_image = call_window_blend(final_image_1, bg_image, window_mask)

    return final_image.astype('uint8')


def random_crop_image(org_image, label, crop_size=(961, 541), threshold=0.8, max_iterations=100):
    h, w = org_image.shape[:2]
    label = label.astype(np.uint8)

    crop_w, crop_h = crop_size
    iters = 0
    while iters < max_iterations:
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        cropped_car_mask = label[top: top + crop_h, left: left + crop_w, 0]  # car mask in 1st channel
        mask_area = np.count_nonzero(cropped_car_mask > 0)

        if mask_area >= crop_h * crop_w * threshold:
            cropped_org_img = org_image[top: top + crop_h, left: left + crop_w]
            cropped_label = label[top: top + crop_h, left: left + crop_w]

            return cropped_org_img, cropped_label
        iters += 1

    return org_image, label


def zoom_in_and_crop(image, label_org, scale_factor, car_threshold_min=0.4, car_threshold_max=0.9, max_iterations=100):
    label = label_org.copy().astype(np.uint8)
    black_channel = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
    label = np.dstack((label, black_channel))

    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    scaled_image = pil_resize(image, (new_width, new_height))
    scaled_label = pil_resize(label, (new_width, new_height))

    crop_w, crop_h = (961, 541)
    iters = 0
    while iters < max_iterations:
        top = random.randint(0, new_height - crop_h)
        left = random.randint(0, new_width - crop_w)
        cropped_car_mask = scaled_label[top: top + crop_h, left: left + crop_w, 0]  # car mask in 1st channel
        car_mask_area_perc = np.count_nonzero(cropped_car_mask > 0) / (crop_h * crop_w)

        if car_mask_area_perc >= car_threshold_min and car_mask_area_perc <= car_threshold_max:
            cropped_org_img = scaled_image[top: top + crop_h, left: left + crop_w]
            cropped_label = scaled_label[top: top + crop_h, left: left + crop_w]
            if cropped_label.shape[2] == 3:
                cropped_label = cropped_label[:, :, :2]

            return cropped_org_img, cropped_label
        iters += 1

    return image, label_org


def find_com(thresh):
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def blend_half_car(original_image, carmask, bg_image):

    carmask_copy = carmask.copy()
    carmask_copy = cv2.cvtColor(carmask_copy, cv2.COLOR_BGR2GRAY)
    _, carmask_copy = cv2.threshold(carmask_copy, 150, 255, cv2.THRESH_BINARY)
    cx, cy = find_com(carmask_copy)

    carmask_mod = carmask.copy()
    h, w = carmask_mod.shape[:2]
    h_offset = int(cy)

    carmask_mod[h_offset:h, :] = 255
    final_image_1 = (original_image * (carmask_mod / 255) + bg_image * (1 - (carmask_mod / 255))).astype("uint8")
    return final_image_1.astype('uint8')


def blur_Mask(mask, kernel=11):
    if len(mask) != 2:
        gray = mask[:, :, 0].copy()
    else:
        gray = mask.copy()

    blur = cv2.pyrUp(gray.astype('uint8'))

    ii = 0
    while ii < 3:
        blur = cv2.medianBlur(blur, kernel)
        ii += 1

    mask_transformed = cv2.pyrDown(blur)

    mask_transformed = cv2.GaussianBlur(mask_transformed, (3, 3), 0)
    mask_transformed = np.dstack((mask_transformed, mask_transformed, mask_transformed)) / 255
    return mask_transformed


def call_window_blend(org_car_image, bg, window_mask):
    final = org_car_image.copy()

    if len(window_mask.shape) == 2:
        window_mask = np.dstack([window_mask, window_mask, window_mask])
    contours, hierarchy = cv2.findContours(window_mask[:, :, 0],
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        window_patch = np.zeros_like(window_mask)
        cv2.drawContours(window_patch, [contour], -1, (255, 255, 255), -1)

        window_patch_dilated = cv2.dilate(window_patch, np.ones((3, 3)))

        window_patch_feathered = np.zeros_like(window_patch_dilated)
        window_patch_feathered[window_patch_dilated > 0] = window_mask[window_patch_dilated > 0]

        mean_values = []
        for i in range(3):
            mean_pixels = final[:, :, i][window_patch[:, :, 0] > 0]
            # print(mean_pixels.shape)
            if len(mean_pixels):
                mean_values.append(np.mean(mean_pixels))
            else:
                mean_values.append(127)

        background_image = np.zeros_like(org_car_image)
        background_image[:, :, 0] = mean_values[0]
        background_image[:, :, 1] = mean_values[1]
        background_image[:, :, 2] = mean_values[2]

        mask_transformed_window = blur_Mask(window_patch_feathered, kernel=3)
        alpha = random.uniform(0.4, 0.7)
        final_wo_window = org_car_image * (1 - mask_transformed_window)  # car blended
        window_blended = mask_transformed_window * background_image * alpha + bg * mask_transformed_window * (1 - alpha)
        org_car_image = (final_wo_window + window_blended).astype("uint8")

    return org_car_image


def modify_bg_image(image, carmask, outside_objects_dir, objects_in_BG):
    rnd_key = random.choice(list(objects_in_BG.keys()))
    objects_in_BG[rnd_key] = True

    if objects_in_BG["logos"]:
        suffix = "_logos_"
        path = os.path.join(outside_objects_dir, "dealers_logo/*")
        diff_logos = np.random.randint(1, 3)  # low inclusive  high: exclusive
        divisor = random.uniform(0.6, 1)


    elif objects_in_BG["small_objects"]:
        suffix = "_small_objects_"
        path = os.path.join(outside_objects_dir, "logos/*")
        diff_logos = np.random.randint(2, 4)
        divisor =  random.uniform(0.2, 0.8)


    else:
        suffix = "_cars_"
        path = os.path.join(outside_objects_dir, "transparent_cropped_car/*")
        diff_logos = np.random.randint(1, 4)
        divisor =  random.uniform(1.2, 1.8)

    mod_image = image.copy()

    try:
        mod_image = paste_objects(mod_image, carmask, path, diff_logos, divisor, objects_in_BG)
    except:
        mod_image = image.copy()

    return mod_image.astype("uint8"), suffix


def pasteimg_pil(transparent_img, background_img, x, y):
    if isinstance(transparent_img, np.ndarray):
        transparent_img = Image.fromarray(transparent_img)

    if isinstance(background_img, np.ndarray):
        background_img = Image.fromarray(background_img)

    # print(f'transparent_img {transparent_img.size}')
    try:
        background_img.paste(transparent_img, (x, y), transparent_img)
    except:
        pass
        # print("return the image as it is")
        # background_img.paste(transparent_img, (x, y))

    return np.array(background_img)


def divide_into_blocks(start, end, num_blocks=3):
    total_length = end - start + 1
    block_size = total_length // num_blocks
    blocks = []

    for i in range(num_blocks):
        block_start = start + i * block_size
        if i == num_blocks - 1:  # Last block should include all remaining values
            block_end = end
        else:
            block_end = block_start + block_size - 1
        blocks.append((block_start, block_end))

    return blocks


def pick_value_less_than_mid(block_start, block_end, upper_x):
    mid = (block_start + block_end) / 2
    q = 0
    while q < 10:
        value = random.randint(block_start, int(mid) - 1)
        q += 1
        if value in upper_x:
            return value

    value = upper_x[1]
    return value


def generate_values(start, end, upper_x, num_blocks):
    blocks = divide_into_blocks(start, end, num_blocks)
    return [pick_value_less_than_mid(block_start, block_end, upper_x) for block_start, block_end in blocks]


def randomize_object(image):
    angle = random.uniform(10, -10)
    w, h = image.shape[1::-1]

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)

    new_w = int(w * abs(np.cos(np.radians(angle))) + h * abs(np.sin(np.radians(angle))))
    new_h = int(w * abs(np.sin(np.radians(angle))) + h * abs(np.cos(np.radians(angle))))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    modified_image = cv2.warpAffine(image, M, (new_w, new_h))

    if random.random() < 0.5:
        modified_image = cv2.flip(modified_image, 1)

    return modified_image


def apply_part(image, logo_np, carmask, num_blocks,objects_in_BG):
    y_coords, x_coords = np.where(cv2.cvtColor(carmask.copy(), cv2.COLOR_BGR2GRAY) > 0)  # Get the uppermost points_,
    _, unique_indices = np.unique(x_coords, return_index=True)

    upper_y = list(y_coords[unique_indices])
    upper_x = list(x_coords[unique_indices])
    cx, cy = find_com(cv2.cvtColor(carmask.copy(), cv2.COLOR_BGR2GRAY))
    filtered_coords = [(upper_x[index], upper_y[index]) for index, y in enumerate(upper_y) if y <= cy]
    upper_x = [x for (x, y) in filtered_coords]
    upper_y = [y for (x, y) in filtered_coords]

    start, end = upper_x[0], upper_x[-1]
    blocks = generate_values(start, end, upper_x, num_blocks=num_blocks)
    pasted_img = image.copy()

    for block in blocks:
        logo_np = randomize_object(logo_np.copy())
        logo_h, logo_w = logo_np.shape[:2]

        if objects_in_BG["cars"]:
            if logo_h > 500 or logo_w > 500:
                ar = logo_w / logo_h
                width = random.randint(300, 500)
                height = int(width / ar)
                logo_np = cv2.resize(logo_np, (width, height))
                # print(f"logo resized from {logo_h, logo_w} to", logo_np.shape[:2])
                logo_h, logo_w = logo_np.shape[:2]

        chosen_value_x = block
        try:
            chosen_value_y = upper_y[upper_x.index(chosen_value_x)] - logo_h // 2
            # print(f"Block: {block}, Chosen Value: {chosen_value_x, chosen_value_y}")
            # print(f"chosen_value_x {chosen_value_x} chosen_value_y {chosen_value_y}")
            # print(f"logo_np {logo_np.shape} pasted_img {pasted_img.size}")
            pasted_img = pasteimg_pil(logo_np, pasted_img, chosen_value_x, chosen_value_y)

        except:
            print("IN EXCEPT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            pasted_img = pasted_img.copy()
            break
    return pasted_img


def obj_transperency(image, object_image):
    if object_image.shape[2] == 4:
        object_h, object_w, c = object_image.shape
        object_resized_obj = cv2.resize(object_image.copy(), (int(object_w * random.randint(1, 4)), object_h))
        object_h, object_w, c = object_resized_obj.shape

        if object_h > image.shape[0]:
            object_h = image.shape[0] // random.randint(2, 3)
            object_resized_obj = cv2.resize(object_resized_obj, (object_w, object_h))

        if object_w > image.shape[1]:
            object_w = image.shape[1] // random.randint(2, 3)
            object_resized_obj = cv2.resize(object_resized_obj, (object_w, object_h))

        bgr = object_resized_obj[..., :3]
        alpha = object_resized_obj[..., 3] / 255.0

        gradient = np.linspace(1, 0, object_resized_obj.shape[0]).reshape(-1, 1)
        transparency_factor = 0.8

        alpha = alpha * transparency_factor * gradient

        x = random.randint(0, image.shape[1])
        y = random.randint(0, (image.shape[0] - random.randint(10, 50)))

        image_1 = image.copy()
        try:
            roi = image[y:y + object_h, x:x + object_w]

            alpha_resized = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))
            bgr_resized = cv2.resize(bgr, (roi.shape[1], roi.shape[0]))

            blended_obj = (1.0 - alpha_resized[:, :, np.newaxis]) * roi + alpha_resized[:, :, np.newaxis] * bgr_resized

            image[y:y + blended_obj.shape[0], x:x + blended_obj.shape[1]] = blended_obj
        except:
            image = image_1.copy()

    return image.astype("uint8")


def glass_transparency(image):
    glass_colour = random.choice([True, False])
    if glass_colour:
        object_glass_image_name_path = "./trans/enhanced_green_image (1).png"
    else:
        object_glass_image_name_path = "./trans/87-873186_glass-texture-png.png"

    object = cv2.imread(object_glass_image_name_path, cv2.IMREAD_UNCHANGED)

    if object.shape[2] == 4:

        object_resized = cv2.resize(object, (image.shape[1], image.shape[0]))

        flip_horizontal = random.choice([True, False])
        flip_vertical = random.choice([True, False])

        if flip_horizontal:
            object_resized = cv2.flip(object_resized, 1)

        if flip_vertical:
            object_resized = cv2.flip(object_resized, 0)

        object_rgb = object_resized[:, :, :3]
        object_alpha = object_resized[:, :, 3] / 255.0

        alpha_mask = np.dstack((object_alpha, object_alpha, object_alpha))

        trans = 0.5
        alpha_mask = alpha_mask * trans

        blended = cv2.convertScaleAbs(image * (1 - alpha_mask) + object_rgb * alpha_mask)

        return blended


def mirror_augmentation(image, object_set_paths):
    object_image = cv2.imread(random.choice(object_set_paths), cv2.IMREAD_UNCHANGED)
    glass_blended = glass_transparency(image)
    glass_blended_WITH_OBJECT = obj_transperency(glass_blended, object_image)
    return glass_blended_WITH_OBJECT


def paste_objects(mod_image, carmask, path, diff_logos, divisor, objects_in_BG):
    for logo in range(diff_logos):
        num_blocks = np.random.randint(2, 4)

        mod_image_in = mod_image.copy()
        allpaths = glob.glob(path)
        logo_path = random.choice(allpaths)
        transparent_img = cv2.imread(logo_path, -1)
        # print("LOGO:", logo_path)

        w, h = transparent_img.shape[1::-1]
        if w > 300 or h >300:
            new_size = (int(w // divisor), int(h // divisor))
        else:
            new_size = (w,h)

        if new_size[0] < carmask.shape[1]  and new_size[1] < carmask.shape[0] :
            transparent_img = cv2.resize(transparent_img, new_size)
            transparent_img = Image.fromarray(transparent_img)
            mod_image = apply_part(mod_image_in, np.array(transparent_img), carmask, num_blocks, objects_in_BG)
        else:
            continue

    return mod_image.astype("uint8")


def modify_org_using_pixels_mean(bg_image, org_image):
    if bg_image.shape != org_image.shape:
        bg_image = cv2.resize(bg_image, (org_image.shape[1], org_image.shape[0]))

    mean_color_image1 = np.mean(bg_image, axis=(0, 1))
    mean_color_image2 = np.mean(org_image, axis=(0, 1))

    adjusted_image2 = org_image + (mean_color_image1 - mean_color_image2)
    adjusted_image2 = np.clip(adjusted_image2, 0, 255).astype(np.uint8)

    return adjusted_image2.astype('uint8')


def paste_half_bg_and_objects(org_img, car_mask, background_img, outside_objects_dir, objects_in_BG, color_matching=False):
    h, w = org_img.shape[:2]
    background_img = pil_resize(background_img, (w, h))
    bg_img_with_objects, suffix = modify_bg_image(background_img, car_mask, outside_objects_dir,
                                                  objects_in_BG)
    if color_matching:
        org_img = modify_org_using_pixels_mean(bg_img_with_objects, org_img)
    image_bg_blended = blend_half_car(org_img, car_mask, bg_img_with_objects)
    return image_bg_blended


def find_side_mirrors_coords(carmask):
    y_coords, x_coords = np.where(cv2.cvtColor(carmask.copy(), cv2.COLOR_BGR2GRAY) > 0)  # Get the uppermost points_,
    _, unique_indices = np.unique(x_coords, return_index=True)

    upper_y = list(y_coords[unique_indices])
    upper_x = list(x_coords[unique_indices])
    cx, cy = find_com(cv2.cvtColor(carmask.copy(), cv2.COLOR_BGR2GRAY))
    filtered_coords = [(upper_x[index], upper_y[index]) for index, y in enumerate(upper_y) if y <= cy]
    upper_x = [x for (x, y) in filtered_coords]
    left_mirror_coords = upper_x[5], int(cy // 2.3)
    right_mirror_coords = upper_x[-5], int(cy // 2.3)
    return left_mirror_coords, right_mirror_coords


def apply_part_mirrors(original_image, label, logo_np, use_main_car=False):
    car_mask, window_mask = get_masks(label)

    touching_edge = check_for_edges(car_mask)
    if touching_edge:
        return original_image

    left_mirror_coords, right_mirror_coords = find_side_mirrors_coords(car_mask)
    mirror_coords = [left_mirror_coords, right_mirror_coords]
    pasted_img = original_image.copy()

    mask = cv2.cvtColor(car_mask.copy(), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        x, y, w, h = 0, 0, mask.shape[1], mask.shape[0]

    if use_main_car:
        cropped_image = original_image[y:y + h, x:x + w]
        four_channel_image = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
        four_channel_image[:, :, :3] = cropped_image
        four_channel_image[:, :, 3] = mask[y:y + h, x:x + w]
        logo_np = four_channel_image

    for mirror_coord in mirror_coords:
        logo_h, logo_w = logo_np.shape[:2]

        ar = logo_w / logo_h
        width = random.randint(w // 3, int(w * 2) // 3)
        height = int(width / ar)

        try:
            logo_np = cv2.resize(logo_np, (width, height))
            chosen_value_x = mirror_coord[0] - int(width / 2)
            chosen_value_y = mirror_coord[1]
            pasted_img = pasteimg_pil(logo_np, pasted_img, chosen_value_x, chosen_value_y)

        except:
            pasted_img = pasted_img.copy()
            break

    car_blended_img = original_image * (car_mask / 255) + pasted_img * (1 - (car_mask / 255))
    # final_img = call_window_blend(car_blended_img, pasted_img, window_mask)

    return car_blended_img.astype("uint8")


def generate_values_partial(numbers):
    ranges = [(0, 240), (240, 480), (480, 620), (620, 960)]
    picked_numbers = []

    for start, end in ranges:
        # Filter numbers within the current range
        numbers_in_range = [n for n in numbers if start <= n < end]
        if numbers_in_range:  # Check if there are any numbers in this range
            picked_numbers.append(random.choice(numbers_in_range))

    return picked_numbers


def get_uppermost_unique_car_points(car_mask_gray):
    # Get the uppermost points
    y_coords, x_coords = np.where(car_mask_gray > 0)
    _, unique_indices = np.unique(x_coords, return_index=True)
    upper_y = list(y_coords[unique_indices])
    upper_x = list(x_coords[unique_indices])
    cx, cy = find_com(car_mask_gray)
    filtered_coords = [(upper_x[index], upper_y[index]) for index, y_value in enumerate(upper_y) if y_value <= cy]
    upper_x = [x_var for (x_var, y_var) in filtered_coords]
    upper_y = [y_var for (x_var, y_var) in filtered_coords]

    return upper_x, upper_y


def shift_points_towards_top_left(block_values, org_shape, logo_shape):
    block_value_x, block_value_y = block_values
    org_h, org_w = org_shape
    logo_h, logo_w = logo_shape

    chosen_value_x = block_value_x - random.randint(block_value_x // 10, block_value_x // 3)
    if int(chosen_value_x + logo_w) > org_w:
        diff_x = (chosen_value_x + logo_w) - org_w
        chosen_value_x = chosen_value_x - diff_x - 5

    chosen_value_y = block_value_y - random.randint(block_value_y // 5, block_value_y // 3)
    if int(chosen_value_y + logo_h) > org_h:
        diff_y = (chosen_value_y + logo_h) - org_h
        chosen_value_y = chosen_value_y - diff_y - 5

    return chosen_value_x, chosen_value_y


def apply_cars_behind_partial_images(original_image, car_mask, logos_paths):
    if len(car_mask.shape) != 2:
        car_mask_gray = cv2.cvtColor(car_mask, cv2.COLOR_BGR2GRAY)
    else:
        car_mask_gray = car_mask

    upper_x, upper_y = get_uppermost_unique_car_points(car_mask_gray)
    blocks_list = generate_values_partial(upper_x)
    if blocks_list == []:
        return original_image

    pasted_img = original_image.copy()
    org_h, org_w = original_image.shape[:2]

    for index, block_value_x in enumerate(blocks_list):
        logo_np = cv2.imread(random.choice(logos_paths), -1)
        logo_np = np.array(Image.fromarray(logo_np))
        logo_np = randomize_object(logo_np) if random.random() < 0.5 else logo_np

        logo_height_org, logo_width_org = logo_np.shape[:2]
        logo_aspect_ratio = logo_width_org / logo_height_org

        # resizing logo based on width of image
        new_width = random.randint(org_w // 6, org_w // 3)
        new_height = int(new_width / logo_aspect_ratio)
        logo_np_final = cv2.resize(logo_np, (new_width, new_height))

        # SHIFTING POINT from CAR EDGE TOWARDS  TOP LEFT SIDE for BETTER PASTING
        block_value_y = upper_y[upper_x.index(block_value_x)]
        chosen_value_x, chosen_value_y = shift_points_towards_top_left(block_values=(block_value_x, block_value_y),
                                                                       org_shape=original_image.shape[:2],
                                                                       logo_shape=logo_np_final.shape[:2])

        pasted_img = pasteimg_pil(logo_np_final, pasted_img, chosen_value_x, chosen_value_y)

    return pasted_img.astype("uint8")


def apply_cars_behind_image(original_image, car_mask, logos_paths, use_main_car=False):
    if len(car_mask.shape) != 2:
        car_mask_gray = cv2.cvtColor(car_mask, cv2.COLOR_BGR2GRAY)
    else:
        car_mask_gray = car_mask

    upper_x, upper_y = get_uppermost_unique_car_points(car_mask_gray)
    blocks_list = generate_values(start=upper_x[0], end=upper_x[-1], upper_x=upper_x, num_blocks=random.randint(1, 4))
    if blocks_list == []:
        return original_image
    # SELECTED EQUIDISTANT blocks , across upper_x values to avoid overlapping and improve pasting

    contours, _ = cv2.findContours(car_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        x, y, w, h = 0, 0, car_mask_gray.shape[1], car_mask_gray.shape[0]

    if use_main_car:
        cropped_image = original_image[y:y + h, x:x + w]
        four_channel_image = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
        four_channel_image[:, :, :3] = cropped_image
        four_channel_image[:, :, 3] = car_mask_gray[y:y + h, x:x + w]
        logo_np_org = four_channel_image

    pasted_img = original_image.copy()

    for index, block_value_x in enumerate(blocks_list):
        if use_main_car:
            logo_np = logo_np_org.copy()

        else:
            logo_np = cv2.imread(random.choice(logos_paths), -1)
            logo_np = np.array(Image.fromarray(logo_np))
            logo_np = randomize_object(logo_np) if random.random() < 0.5 else logo_np

        logo_height_org, logo_width_org = logo_np.shape[:2]
        logo_aspect_ratio = logo_width_org / logo_height_org

        # resizing logo based on width of car bbox
        new_width = random.randint(w // 5, w // 2)
        new_height = int(new_width / logo_aspect_ratio)
        logo_np_final = cv2.resize(logo_np, (new_width, new_height))

        # SHIFTING POINT from CAR EDGE TOWARDS  TOP LEFT SIDE for BETTER PASTING
        block_value_y = upper_y[upper_x.index(block_value_x)]
        chosen_value_x, chosen_value_y = shift_points_towards_top_left(block_values=(block_value_x, block_value_y),
                                                                       org_shape=original_image.shape[:2],
                                                                       logo_shape=logo_np_final.shape[:2])

        pasted_img = pasteimg_pil(logo_np_final, pasted_img, chosen_value_x, chosen_value_y)

    return pasted_img.astype("uint8")


def center_crop(image, label):
    contours, _ = cv2.findContours(label[:, :, 0].astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    #
    # touching_edge = check_for_edges(label[:, :, 0].astype("uint8"))
    # if touching_edge:
    #     # print("======" * 10)
    #     return image, label

    image = Image.fromarray(image)
    label = Image.fromarray(label.astype("uint8"))

    bbox = [x, y, x + w, y + h]
    # Calculate the center of the bounding box
    bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    # Calculate the size of the bounding box
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    # Desired aspect ratio, e.g., 16:9, 4:3, 1:1 (square)
    aspect_ratio = 16 / 9

    # padd_perc = (random.randint(5, 12)) / 100
    padd_perc = random.uniform(.05, .12)

    # Calculate the maximum crop size without cutting the car
    crop_width = max(bbox_width, bbox_height * aspect_ratio)
    crop_height = max(bbox_height, bbox_width / aspect_ratio)
    crop_width = crop_width + crop_width * padd_perc
    crop_height = crop_height + crop_height * padd_perc

    # Adjust crop size to fit within the image dimensions
    crop_width = min(crop_width, image.width)
    crop_height = min(crop_height, image.height)

    # Calculate the top-left corner of the crop area
    crop_x = max(0, bbox_center[0] - crop_width // 2)
    crop_y = max(0, bbox_center[1] - crop_height // 2)

    # Adjust crop area to fit within the image
    crop_x = min(crop_x, image.width - crop_width)
    crop_y = min(crop_y, image.height - crop_height)

    # Crop the image
    cropped_image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    cropped_label = label.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

    return np.array(pil_resize(cropped_image, (961, 541))), np.array(
        pil_resize_label(np.array(cropped_label), (961, 541)))


def shadow_blend(mod_image, label, image_name, shadow_img_dir, shadow_white_dir):
    image_name = os.path.basename(image_name)
    if "CON_" in image_name:
        splits = image_name.split("_")
        splits = splits[2:-2]
        image_name = "_".join(splits) + ".jpg"

    if "__COPY__" in image_name:
        image_name = image_name.split("__COPY__")[0] + ".jpg"
        # print("MODIFIED >>>>>>>>>>>>>>")

    if "pose" in image_name:
        image_name = image_name.split("_pose")[0] + ".jpg"

    if "BOOTH_" in image_name:
        image_name = image_name.replace("BOOTH_", "")

    png_name = image_name.replace(".jpg", ".png")

    if not os.path.exists(os.path.join(shadow_img_dir, png_name)):
        return mod_image

    shadow_image = cv2.imread(os.path.join(shadow_img_dir, png_name))
    shadow_white = cv2.imread(os.path.join(shadow_white_dir, png_name))

    car_mask, _ = get_masks(label)
    image_transformed = mod_image.copy()

    final = (image_transformed * (1 - (shadow_white / 255)) + shadow_image * shadow_white / 255).astype('uint8')
    final = (image_transformed * (car_mask / 255) + final * (1 - (car_mask / 255))).astype('uint8')

    return final.astype("uint8")


def check_for_edges(input_car_mask):
    car_mask = input_car_mask
    if len(input_car_mask.shape) != 2:
        car_mask = cv2.cvtColor(input_car_mask, cv2.COLOR_BGR2GRAY)

    height, width = car_mask.shape
    margin = 3

    # Check if the car is touching the edges within the last 5 pixels in all directions
    touching_top = np.any(car_mask[:margin, :])  # Check the top margin
    touching_bottom = np.any(car_mask[height - margin:, :])  # Check the bottom margin
    touching_left = np.any(car_mask[:, :margin])  # Check the left margin
    touching_right = np.any(car_mask[:, width - margin:])  # Check the right margin

    if touching_top or touching_bottom or touching_left or touching_right:
        return True
    else:
        return False


def do_mod_lines(image, car_mask):
    org_h, org_w = image.shape[:2]
    contour, _ = cv2.findContours(car_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contour, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    num_lines = random.randint(5, 15)

    mod_image = image.copy()
    for i in range(num_lines):
        thickness = random.randint(3, 12)
        r = random.randint(0, 100)
        g = random.randint(0, 100)
        b = random.randint(0, 100)

        x = max(0, x - 25)
        y = max(0, y - 25)
        w = min(w + 25, org_w)
        h = min(h + 25, org_h)

        picker = random.random()
        if picker < 0.3:
            x1 = random.randint(x, x + w)
            x2 = random.randint(x, x + w)
            y1 = random.randint(y, y + h)
            y2 = random.randint(y, y + h)
        elif picker < 0.65:
            x1 = 0
            x2 = org_w
            y1 = random.randint(y, y + h)
            y2 = random.randint(y, y + h)
        else:
            x1 = random.randint(x, x + w)
            x2 = random.randint(x, x + w)
            y1 = 0
            y2 = org_h

        mod_image = cv2.line(mod_image, (x1, y1), (x2, y2), (b, g, r), thickness)

    return mod_image


def do_mod_vertical_lines(image,  car_mask):
    org_h, org_w = image.shape[:2]
    contour, _ = cv2.findContours(car_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contour, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    num_lines = random.randint(5, 15)
    mod_image = image.copy()


    color = random.randint(50, 220)
    r, g, b = color, color, color

    for i in range(num_lines):
        thickness = 1

        x = max(0, x - 25)
        y = max(0, y - 25)
        w = min(w + 25, org_w)
        h = min(h + 25, org_h)

        picker = random.random()
        if picker <= 0.5:  # pure vertical
            x1 = random.randint(x, x + w)
            y1 = random.randint(0, y)
            x2 = x1
            y2 = random.randint(0, y1 + h // 2)
        else:  # slant
            x1 = random.randint(x, x + w)
            y1 = random.randint(0, y)
            x2 = random.randint(x1 - 30, x1 + 30)
            y2 = random.randint(y1, y1 + h // 2)


        mod_image = cv2.line(mod_image, (x1, y1), (x2, y2), (r, g, b), thickness)

    return mod_image


def do_mod_shapes(image,  car_mask):
    mod_image = image.copy()
    org_h, org_w = image.shape[:2]
    contour, _ = cv2.findContours(car_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contour, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    number_of_drawing = random.randint(1, 4)
    for i in range(number_of_drawing):

        thickness = random.randint(3, 12)
        r = random.randint(0, 100)
        g = random.randint(0, 100)
        b = random.randint(0, 100)

        shape = random.choice(['rectangle', 'circle', 'ellipse'])
        if shape == 'rectangle':
            st_x = random.randint(x - 20, x) if x > 20 else x
            st_y = random.randint(y - 20, y) if y > 20 else y

            end_x = st_x + w if st_x + w < org_w - 2 else org_w - 2
            end_y = st_y + w if st_y + h < org_h - 2 else org_h - 2

            mod_image = cv2.rectangle(mod_image, (st_x, st_y), (end_x, end_y), (b, g, r), thickness)
        elif shape == 'circle':
            cx = x + w // 2 + random.randint(-w // 10, w // 10)
            cy = y + h // 2 + random.randint(-h // 10, h // 10)

            radius = random.randint(min(w // 2, h // 2), max(w // 2, h // 2))
            mod_image = cv2.circle(mod_image, (cx, cy), radius, (b, g, r), thickness)
        elif shape == 'ellipse':
            cx = x + w // 2 + random.randint(-w // 10, w // 10)
            cy = y + h // 2 + random.randint(-h // 10, h // 10)

            axis_len_x = random.randint(w // 2 - 25, w // 2 + 25)
            axis_len_y = random.randint(h // 2 - 25, h // 2 + 25)

            rotation = random.randint(-30, 30)
            mod_image = cv2.ellipse(mod_image, (cx, cy), (axis_len_x, axis_len_y), rotation, 0, 360, (b, g, r),
                                    thickness)
    return mod_image


def do_mod_words(image,  car_mask):
    mod_image = image.copy()
    contour, _ = cv2.findContours(car_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contour, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    num_lines = random.randint(3, 6)

    line_num = 0
    while line_num < num_lines:
        length_of_string = random.randint(10, w // 20) if w // 20 > 10 else 10
        string_of_char = ''.join(random.choice(string.ascii_letters) for _ in range(length_of_string))

        thickness = random.randint(1, 8)
        r = random.randint(0, 100)
        g = random.randint(0, 100)
        b = random.randint(0, 100)

        stx = x - min(0, x - w // 4)
        sty = y + int((line_num / num_lines) * h) + 2

        font = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 16])
        scale = random.randint(1, 3)

        try:
            mod_image = cv2.putText(mod_image, string_of_char, (stx, sty), font, scale, (b, g, r), thickness)
        except cv2.error as e:
            # print(e)
            # print(font)
            pass
        line_num += 1

    return mod_image







