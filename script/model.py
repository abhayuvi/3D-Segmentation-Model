class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.vnet = VNet(spatial_dims=3, in_channels=1, out_channels=4)  # 4 classes: background, liver, kidneys, spleen
    
    def forward(self, x):
        return self.vnet(x)
