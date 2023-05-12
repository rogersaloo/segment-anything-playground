import torch
from official_sam_repo.segment_anything import sam_model_registry
from torch.nn.functional import threshold, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download SAM model checkpoint
sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')

# Use the Adam Optimizer as default optimizer to use on the mask decorder only
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())

# Set a loss function of MSE
loss_fn = torch.nn.MSELoss()


with torch.no_grad():
	image_embedding = sam_model.image_encoder(input_image)

with torch.no_grad():
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

low_res_masks, iou_predictions = sam_model.mask_decoder(
  image_embeddings=image_embedding,
  image_pe=sam_model.prompt_encoder.get_dense_pe(),
  sparse_prompt_embeddings=sparse_embeddings,
  dense_prompt_embeddings=dense_embeddings,
  multimask_output=False,
)


# Upscale mask to the orginal size
upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)

binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)

#calculate loss and run optimization step
loss = loss_fn(binary_mask, gt_binary_mask)
optimizer.zero_grad()
loss.backward()
optimizer.step()

torch.save(model.state_dict(), PATH)
