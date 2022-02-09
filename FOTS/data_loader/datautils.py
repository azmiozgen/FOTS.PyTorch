import cv2
import numpy as np
import torch

def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    Parameters
    ----------
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    Returns
    -------
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    """

    ar_ = (get_bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    delta_area = ((ar_ - get_bbox_area(bbox))/ar_)
    mask = (delta_area < (1 - alpha)).astype(int)
    bbox = bbox[mask == 1,:]

    return bbox

def collate_fn(batch):
    image_files, image, score_map, training_mask, transcriptions, boxes = zip(*batch)
    bs = len(score_map)
    images = []
    score_maps = []
    training_masks = []

    for i in range(bs):
        if image[i] is not None:
            a = torch.from_numpy(image[i])
            a = a.permute(2, 0, 1)
            images.append(a)
            b = torch.from_numpy(score_map[i])
            b = b.permute(2, 0, 1)
            score_maps.append(b)
            c = torch.from_numpy(training_mask[i])
            c = c.permute(2, 0, 1)
            training_masks.append(c)

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    mapping = np.arange(len(transcriptions))
    bboxes = np.stack(boxes, axis=0)
    transcriptions = np.stack(transcriptions).flatten()
    bboxes = np.concatenate([bboxes, np.ones((len(bboxes), 1))], axis=1).astype(np.float32)

    return image_files, images, score_maps, training_masks, transcriptions, bboxes, mapping

def denormalize(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=np.float32)
    return to_min + (scaled * to_range)

def denormalize_tensor(tensor, from_min, from_max, to_min, to_max):
    '''
    tensor: tensor of shape (B, C, H, W)
    '''
    for i in range(tensor.size(0)):
        image = np.array(tensor[i])
        image = denormalize(image, from_min, from_max, to_min, to_max)
        tensor[i] = torch.from_numpy(image)
    return tensor

def draw_text_image(text,
            size=128,
            font=cv2.FONT_HERSHEY_PLAIN,
            font_scale=2,
            color=(0, 0, 0),  ## Black text
            thickness=2,
            line_type=cv2.LINE_AA):
    canvas = np.ones((size, size, 3), dtype=np.int32) * 255  ## White canvas
    text_size, _ = cv2.getTextSize(text=text, fontFace=font, fontScale=font_scale, thickness=thickness)
    text_width, text_height = text_size
    canvas = cv2.putText(canvas,
            text,
            (size // 2 - text_width // 2, size // 2 + text_height // 2),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=line_type)
    return canvas

def draw_text_tensor(text_list,
            size=128,
            font=cv2.FONT_HERSHEY_PLAIN,
            font_scale=2,
            color=(0, 0, 0),
            thickness=2,
            line_type=cv2.LINE_AA):
    '''
    tensor: tensor of shape
    '''
    b = len(text_list)
    tensor = torch.zeros((b, 3, size, size))  ## (B, C, H, W)
    for i in range(b):
        canvas = draw_text_image(text_list[i], size, font, font_scale, color, thickness, line_type)
        canvas = canvas.transpose(2, 0, 1)
        tensor[i] = torch.from_numpy(canvas)
    return tensor

def get_bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def get_corners(bboxes):
    """Get corners of bounding boxes
    Parameters
    ----------
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    returns
    -------
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1 
    x3 = x1
    y3 = y1 + height
    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)
    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners

def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

def is_bbox_ok(bbox, length=4):
    if len(bbox) != length:
        return False

    if not all([b >= 0 for b in bbox]):
        return False

    x1, y1, x2, y2 = bbox
    if not ((x1 < x2) and (y1 < y2)):
        return False

    return True

def is_bbox8_ok(bbox, length=8):
    if len(bbox) != length:
        return False

    if not all([b >= 0 for b in bbox]):
        return False

    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    condition1 = (x1 < x2) and (x2 == x3) and (x3 > x4) and (x4 == x1)
    condition2 = (y1 == y2) and (y2 < y3) and (y3 == y4) and (y4 > y1)
    if not (condition1 and condition2):
        return False

    return True

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding
    Parameters
    ----------
    img : numpy.ndarray
        Image 
    inp_dim: tuple(int)
        shape of the reszied image
    Returns
    -------
    numpy.ndarray:
        Resized image
    '''

    inp_dim = (inp_dim, inp_dim)
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / (img_w + 1e-10), h / (img_h + 1e-10)))
    new_h = int(img_h * min(w / (img_w + 1e-10), h / (img_h + 1e-10)))
    resized_image = cv2.resize(img, (new_w, new_h))
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)

    canvas[(h - new_h) // 2 : (h - new_h) // 2 + new_h, \
           (w - new_w) // 2 : (w - new_w) // 2 + new_w, :] = resized_image
    return canvas

def resize(image, bbox, size, interpolation=cv2.INTER_NEAREST):
    _h, _w, _ = image.shape
    h, w = size
    image = cv2.resize(image, dsize=(h, w), interpolation=interpolation)
    ratio_h = h / _h
    ratio_w = w / _w
    bbox *= np.array([ratio_w, ratio_h, ratio_w, ratio_h])
    return image, bbox

def rotate_image(image, angle):
    """Rotate the image.
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    Parameters
    ----------
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    Returns
    -------
    numpy.ndarray
        Rotated Image
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image

def rotate_box(corners,angle,  cx, cy, h, w):
    """Rotate the bounding box.
    Parameters
    ----------
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    angle : float
        angle by which the image is to be rotated
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
    h : int 
        height of the image
    w : int 
        width of the image
    Returns
    -------
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T

    calculated = calculated.reshape(-1,8)

    return calculated
