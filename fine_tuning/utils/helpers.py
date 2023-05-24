import numpy as np


class Helper:

    @classmethod
    def show_mask(mask, ax, random_color=False):
        """_summary_

        Args:
            mask (_type_): _description_
            ax (_type_): _description_
            random_color (bool, optional): _description_. Defaults to False.
        """
    
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    @classmethod
    def show_box(box ,
                 ax):
        """Helper function for bounding box image

        Args:
            box (_type_): _description_
            ax (_type_): _description_
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))



# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
