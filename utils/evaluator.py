import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
from tqdm import tqdm



'''labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]'''


class SemanticSegmentationEvaluator:
    def __init__(self, directory, classes=None, class_names=None, void_color=(0, 0, 0), mode='rgb'):
        """
        Initializes the evaluator with the directory containing subfolders for predictions and ground_truth.
        
        Args:
            directory (str): Path to the directory containing subfolders for 'ground_truth' and various predictions.
            void_color (tuple): RGB color representing the void class for invalid predictions in RGB mode.
        """
        self.directory = directory
        self.gt_dir = os.path.join(directory, 'ground_truth')
        self.void_color = void_color
        self.classes = classes # Will be determined from ground truth images if None
        self.class_names = class_names # Will be determined from ground truth images if None
        self.class_dict = None  # Maps RGB to class index
        if self.classes is not None: 
            self.class_dict = {color: idx for idx, color in enumerate(self.classes)}
        self.mode = mode  # Will be set to 'rgb' or 'grayscale' depending on the image format

    def _extract_unique_classes_from_gt(self):
        """
        Extract unique classes from ground truth images only, whether they are in RGB or grayscale.
        """
        unique_classes = set()
            
        # Valid image file extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        # Get a list of image files in the directory
        filenames = [f for f in os.listdir(self.gt_dir) if f.lower().endswith(valid_extensions)]

        for filename in tqdm(filenames, desc="Extracting classes from ground truth"):
            image = Image.open(os.path.join(self.gt_dir, filename))

            if image.mode == 'RGB':
                img_np = np.array(image)
                self.mode = 'rgb'
                # Get all unique RGB values in the ground truth image and add them to the set
                unique_colors = np.unique(img_np.reshape(-1, img_np.shape[2]), axis=0)
                unique_classes.update(map(tuple, unique_colors))  # Convert to tuple to add to set
            elif image.mode == 'L':  # Grayscale image
                img_np = np.array(image)
                self.mode = 'grayscale'
                self.void_color = 0
                # Get all unique grayscale values (class indices)
                unique_classes.update(np.unique(img_np))

                # Convert set to sorted list and create a dictionary for fast lookup (for RGB mode)
                self.classes = sorted(list(unique_classes))
                self.classes  = [(int(r), int(g), int(b)) for (r, g, b) in self.classes]
            
        print(f"Discovered classes from ground truth: {self.classes}")

    def _rgb_to_class(self, img_np):
        """
        Converts an RGB or grayscale image into class indices based on discovered classes.
        
        Args:
            img_np (np.ndarray): RGB or grayscale image as a NumPy array.

        Returns:
            np.ndarray: Class indices.
        """
        #print(f'SELF MODE {self.mode}')
        if self.mode == 'grayscale':
            # In grayscale mode, the pixel values are already the class indices
            return img_np

        elif self.mode == 'rgb':
            # Vectorized mapping from RGB to class index using the class dictionary
            class_map = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

            # Convert every pixel to its class index
            for color, idx in self.class_dict.items():
                mask = np.all(img_np == color, axis=-1)
                class_map[mask] = idx

            return class_map

    def _load_images(self, gt_filename, pred_filename, pred_dir):
        """
        Load the predicted and ground truth images.

        Args:
            gt_filename (str): Filename from the ground truth folder.
            pred_filename (str): Filename from the prediction folder (should match ground truth filename).
            pred_dir (str): Path to the current prediction directory.

        Returns:
            tuple: Tuple of (predicted_class_image, ground_truth_class_image).
        """
        # Load ground truth image
        gt_image = Image.open(os.path.join(self.gt_dir, gt_filename))
        gt_img_np = np.array(gt_image)
        #print(f'gt_img_np shape: {gt_img_np.shape}, path: {os.path.join(self.gt_dir, gt_filename)} ')

        # Load predicted image
        pred_image = Image.open(os.path.join(pred_dir, pred_filename))
        pred_img_np = np.array(pred_image)
        #print(f'pred_img_np shape: {pred_img_np.shape}, path: {os.path.join(pred_dir, pred_filename)}')

        # Convert images to class maps based on the mode (RGB or Grayscale)
        gt_class = self._rgb_to_class(gt_img_np)
        pred_class = self._rgb_to_class(pred_img_np)

        return pred_class, gt_class
    
    # TODO: Update to calc per category IIF categories are present
    def _calculate_iou_dice(self, conf_matrix):
        """
        Calculate IoU and Dice coefficient for each class given a confusion matrix.
        
        :param conf_matrix: Confusion matrix
        :return: IoU and Dice coefficient per class, average IoU, and average Dice
        """
        IoU_per_class = []
        dice_per_class = []
        
        # For each class, calculate IoU and Dice
        for i in range(len(conf_matrix)):
            TP = conf_matrix[i, i]  # True Positives for class i
            FP = conf_matrix[:, i].sum() - TP  # False Positives for class i
            FN = conf_matrix[i, :].sum() - TP  # False Negatives for class i
            
            denominator = TP + FP + FN
            
            # IoU
            IoU = TP / denominator if denominator > 0 else 0
            IoU_per_class.append(IoU)
            
            # Dice coefficient
            Dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            dice_per_class.append(Dice)
        
        # Calculate average IoU and Dice coefficient
        avg_IoU = np.mean(IoU_per_class)
        avg_dice = np.mean(dice_per_class)
        
        return IoU_per_class, dice_per_class, avg_IoU, avg_dice

    def evaluate(self):
        """
        Evaluate the predictions from each model against the ground truth and compute the confusion matrix and accuracy.

        Returns:
            dict: A dictionary containing confusion matrices and accuracies for each model.
        """
        all_model_results = {}

        # Step 1: Extract the unique classes from the ground truth images
        if self.classes is None or len(self.classes) < 1:
            self._extract_unique_classes_from_gt()
            self.class_names = [f'C{x}' for x in range(1, len(self.classes))]
            self.class_dict = {color: idx for idx, color in enumerate(self.classes)}

        print(f"Final classes (excluding void): {self.classes}")

        # Step 3: Identify all prediction folders (excluding 'ground_truth')
        all_subdirs = [d for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, d)) and d != 'ground_truth']

        # Step 4: Process each prediction folder (model)
        for pred_folder in all_subdirs:
            print(f"Evaluating predictions from: {pred_folder}")
            pred_dir = os.path.join(self.directory, pred_folder)

            all_preds = []
            all_gts = []
            
            # Valid image file extensions
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

            # Get a list of image files in the directory
            filenames = [f for f in os.listdir(self.gt_dir) if f.lower().endswith(valid_extensions)]

            # Process images for the current prediction folder with progress bar
            for filename in tqdm(filenames, desc=f"Processing images for {pred_folder}"):
                pred_class, gt_class = self._load_images(filename, filename, pred_dir)
                all_preds.append(pred_class.flatten())
                all_gts.append(gt_class.flatten())
                
            # Flatten all predictions and ground truths for confusion matrix calculation
            all_preds = np.concatenate(all_preds)
            all_gts = np.concatenate(all_gts)

            # Compute the confusion matrix and accuracy for the current prediction folder
            conf_matrix = confusion_matrix(all_gts, all_preds, labels=range(1, len(self.classes))) # Ignore void values
            #print(conf_matrix)
            accuracy = accuracy_score(all_gts, all_preds)

            # Calculate IoU and Dice coefficient per class and overall
            IoU_per_class, dice_per_class, avg_IoU, avg_dice = self._calculate_iou_dice(conf_matrix)

            # Store the results for this prediction folder
            all_model_results[pred_folder] = {
                'confusion_matrix': conf_matrix.tolist(),
                'accuracy': accuracy,
                'IoU_per_class': IoU_per_class,
                'dice_per_class': dice_per_class,
                'avg_IoU': avg_IoU,
                'avg_dice': avg_dice,
                'class_names': self.class_names
            }
        return all_model_results

# Example usage:
if __name__ == "__main__":
    evaluator = SemanticSegmentationEvaluator(directory="/Users/noellelaw/Desktop/visualizations/triangles/uploads/zip/triangles", void_color=(0, 0, 0))
    results = evaluator.evaluate()

    # Display results for each model
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"Overall Accuracy: {metrics['accuracy']:.2f}")
        print("-" * 40)

