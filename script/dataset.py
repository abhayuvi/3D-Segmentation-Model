class CTScanDataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_depth=None, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = self._get_valid_files(images_dir)
        self.label_files = self._get_valid_files(labels_dir)

        if target_depth is None:
            self.target_depth = self._determine_max_depth()
        else:
            self.target_depth = target_depth

    def _get_valid_files(self, directory):
        # Get only valid files, ignoring .DS_Store and other non-image files
        return sorted([f for f in os.listdir(directory) if not f.startswith('.') and f.endswith('.nii')])

    def _determine_max_depth(self):
        max_depth = 0
        for img_file in self.image_files:
            img_path = os.path.join(self.images_dir, img_file)
            img = nib.load(img_path).get_fdata()
            depth = img.shape[-1]
            if depth > max_depth:
                max_depth = depth
        return max_depth

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        # Load the image and label using nibabel
        image = nib.load(image_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.float32)

        # Convert to PyTorch tensors
        image = torch.tensor(image).unsqueeze(0)  # [1, H, W, D]
        label = torch.tensor(label).unsqueeze(0)  # [1, H, W, D]

        # Calculate the padding needed
        depth_diff = self.target_depth - image.shape[-1]
        if depth_diff > 0:
            image = torch.nn.functional.pad(image, (0, depth_diff), mode='constant', value=0)
            label = torch.nn.functional.pad(label, (0, depth_diff), mode='constant', value=0)
        elif depth_diff < 0:
            image = image[:, :, :, :self.target_depth]
            label = label[:, :, :, :self.target_depth]

        if self.transform:
            image, label = self.transform(image, label)

        return image, label
