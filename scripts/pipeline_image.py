import torch
import numpy as np
import os
from torchvision.utils import save_image
from PIL import Image
# from models import fluid # video_fluid_arbitrary
from models import fluid_arbitrary as fluid
from models.utils import T5_Embedding
import argparse
from pathlib import Path
from diffusers.models import AutoencoderKL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # init the model architecture
    # img_size = 256
    vae_stride = 8
    patch_size = 2
    diffloss_d = 8
    diffloss_w = 1536
    num_sampling_steps = '100'
    model_type = "fluid_large"


    model_checkpoint_path = "fluid_dpo/checkpoint-20.pth"
    # model_checkpoint_path = '/data/xianfeng/code/mar/fluid_cache_1024/checkpoint-last.pth'

    model = fluid.__dict__[model_type](
        vae_stride=vae_stride,
        patch_size=patch_size,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        text_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        diffloss_d=diffloss_d,
        diffloss_w=diffloss_w,
        max_length=128,
        num_sampling_steps=num_sampling_steps,
    ).cuda()

    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.cuda()

    # download and load the vae
    vae_checkpoint_path = '/data/xianfeng/code/model/stabilityai/stable-diffusion-3.5-large'

    vae = AutoencoderKL.from_pretrained(os.path.join(vae_checkpoint_path, "vae")).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    max_length = 128
    mt5_cache_dir = '/data/xianfeng/code/model/google/flan-t5-xxl'
    t5_emb = T5_Embedding(mt5_cache_dir, mt5_cache_dir, max_length).cuda()

    # set up user-specified or default values for generation
    seed = 4042 # 1024 # 24
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_ar_steps = 64
    cfg_scale = 4
    cfg_schedule = "constant"
    temperature = 1.0
    # text_prompt = [
    #     "A rustic wooden bridge crosses over a calm river surrounded by lush greenery and trees with autumn-colored leaves.",
    #     "A set of three ancient bronze artifacts, each with a different design, are displayed against a white background. The artifacts are positioned vertically, with the leftmost one being the shortest and the rightmost one being the tallest. Each artifact has a distinct shape and pattern, with the leftmost one featuring a simple, flat design, the middle one having a more intricate pattern with a circular element, and the rightmost one showcasing a more complex design with multiple circular elements and a bow-like structure. The artifacts are accompanied by a ruler with centimeter markings, indicating their size.",
    #     "A close-up image of a white flower with a yellow center, surrounded by green leaves. The flower has a textured appearance with visible veins and a soft, delicate look. The background is out of focus, emphasizing the flower's details.",
    #     "A beautiful waterfall."
    #     ]
    # text_prompt = [
    #         "A serene mountain lake surrounded by lush green forests under a clear blue sky.",
    #         "A golden sunset over a sandy beach with gentle waves and scattered seashells.",
    #         "A snowy mountain peak with a clear path leading through a pine forest.", 
    #     ]
    
    # text_prompt = [
    #     "A giant panda wearing a wizard's hat and robe, riding a flying carpet above a magical forest with glowing trees and sparkling stars in the night sky.",
    #     "A unicorn with rainbow-colored fur and a golden horn, soaring through the sky while holding a treasure chest with its hooves, with fluffy clouds and a bright rainbow in the background.", 
    #     "A robot with a flower garden on its head, floating in space among colorful planets and stars, with beams of light connecting the planets."
    # ]

    # text_prompt = [
    #     "A mystical phoenix with shimmering, iridescent wings, soaring over a reflective enchanted lake beneath a moonlit sky filled with twinkling stars.",
    #     "An ethereal dragon with translucent scales and glowing eyes, gliding through a surreal sky swirling with nebulae and sparkling constellations.",
    #     "A cybernetic mermaid with holographic fins, navigating a futuristic underwater city of glass and light, surrounded by bioluminescent marine life.",
    #     "A steampunk griffin with bronze wings and intricate clockwork gears, perched atop a floating island amid cascading waterfalls and drifting magical mist.",
    #     "A celestial fox with a tail of sparkling stardust, leaping over a dreamlike landscape of floating crystal formations and pulsating energy orbs.",
    #     "A robotic samurai in neon-lit armor, standing in a surreal garden of bioluminescent flowers and holographic cherry blossoms under a cosmic sky."
    # ]

    # text_prompt = [
    #     "A high-resolution photograph of a majestic lion resting in the golden savannah during sunset, with detailed fur texture and natural lighting.",
    #     "A close-up shot of a brown bear in a dense forest, with realistic details and a calm expression, captured in soft, natural light.",
    #     "A vivid, lifelike image of a red fox in a snowy environment, with its bushy tail and alert eyes, against a serene winter background.",
    #     "A detailed portrait of a wise old elephant in its natural habitat, showcasing its textured skin and gentle eyes in a warm, realistic setting.",
    #     "A clear, natural photograph of a penguin on an icy shoreline, with realistic reflections and crisp details under a clear blue sky.",
    #     "A dynamic, realistic image of a bald eagle in mid-flight, with outstretched wings and sharp gaze, captured over a rugged mountain landscape."
    # ]

    # text_prompt = [
    #     "A high-resolution photograph of a mountain range with snow-capped peaks and a serene lake in the foreground, bathed in soft morning light.",
    #     "A detailed, realistic image of a lush green valley with a winding river and scattered wildflowers under a bright blue sky.",
    #     "A panoramic view of a coastal landscape, with rocky cliffs, crashing waves, and a dramatic sunset painting the sky in warm hues.",
    #     "A serene countryside scene with rolling hills, a clear blue sky, and a rustic farmhouse surrounded by blooming trees.",
    #     "A realistic depiction of a dense forest during autumn, with vibrant foliage, sun rays piercing through the canopy, and a carpet of fallen leaves.",
    #     "A detailed landscape photograph of a desert with vast sand dunes, clear skies, and a solitary cactus standing tall in the distance."
    # ]

    # text_prompt = [
    #     "An angry duck doing heavy weightlifting at the gym.", 
    #     "A large green truck on a city street.", 
    #     "a corgi wearing a red bowtie and a purple party hat", 
    # ]

    # text_prompt = [
    #     'A grand library with ornate wooden ceilings, red carpeted stairs, and shelves filled with books on both sides. The warm lighting highlights the intricate details of the architecture.',
    #     'A monk in orange robes sits on a plastic chair, skillfully weaving with a loom, surrounded by beams of sunlight filtering through the trees.', 
    #     'A close-up black-and-white portrait of a person wearing a headscarf, with dramatic makeup accentuating the eyes and lips. The background is dark, emphasizing the subject\'s features.'
    # ]

    # text_prompt = [ # cherrypick
    #     'A man with short, spiky hair is seated in a dark leather chair, wearing a denim jacket over a black shirt. Behind him, there is a framed black-and-white photograph of a car on the wall.',
    #     'A woman with curly red hair stands in front of a bush of white roses, her hand gently touching the flowers. The scene is serene and natural, with soft lighting enhancing the delicate beauty of the roses and the subject\'s gentle expression.',
    #     'A close-up black-and-white portrait of a person wearing a headscarf, with dramatic makeup accentuating the eyes and lips. The background is dark, emphasizing the subject\'s features.',
    #     'A woman stands on a red carpet, wearing a strapless, form-fitting, taupe-colored dress that accentuates her figure. Her hair is styled in loose waves, and she wears a small, delicate earring. Behind her, other attendees and a hedge line the background.'
    # ]

    text_prompt = [
        'A woman stands on a city street, wearing an elegant white lace dress with intricate patterns and a deep neckline. Her long, wavy hair frames her face as she gazes into the distance.', 
        'A close-up portrait of a person with long, wavy blonde hair styled in soft curls. The individual is wearing bold red lipstick and dramatic eye makeup, including dark eyeliner and mascara. The background is a neutral gray, emphasizing the subject\'s features.', 
        'A man in a top hat and formal attire stands confidently in front of a large, rusted chain-link fence. The chains are thick and interlocked, creating a textured backdrop. The image has a vintage, sepia-toned quality.', 
        'A light-colored bear stands on a moss-covered rock amidst a flowing river, surrounded by lush greenery and vibrant foliage.', 
        'A group of white horses gallops through shallow water under a dramatic sky, creating splashes as they move. The scene is bathed in warm, golden light, highlighting the horses\' graceful motion.', 
        'A fisherman sits on a bamboo raft by the riverbank, holding a fishing net, with a lantern hanging nearby. A cormorant stands on the water behind him, and the backdrop features towering karst mountains under a warm, sunset sky.'
    ]

    text_emb = t5_emb(text_prompt)

    # generate the tokens and images
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            sampled_tokens = model.sample_tokens(
                bsz=len(text_prompt), num_iter=num_ar_steps,
                cfg=cfg_scale, cfg_schedule=cfg_schedule,
                texts=text_emb, temperature=temperature, 
                height=512, width=512, progress=True,
            )
            # if (hasattr(vae.config, "shift_factor") and vae.config.shift_factor):
            #     sampled_tokens = (sampled_tokens / vae.config.scaling_factor + vae.config.shift_factor)
            # else:
            #     sampled_tokens = sampled_tokens / vae.config.scaling_factor
            # sampled_images = vae.decode(sampled_tokens/vae.config.scaling_factor, return_dict=False)[0]
            # output_samples = sampled_images.squeeze()
            if vae.config.shift_factor is not None:
                samples = sampled_tokens / vae.config.scaling_factor + vae.config.shift_factor
            else:
                samples = sampled_tokens / vae.config.scaling_factor   
            output_samples = vae.decode(samples).sample

    output_dir = "output_512_dpo_cherry"
    os.makedirs(output_dir, exist_ok=True)

    # save the images
    image_path = os.path.join(output_dir, "sampled_image.png")
    samples_per_row = 3

    save_image(
        output_samples, image_path, nrow=int(samples_per_row), normalize=True, value_range=(-1, 1)
    )
    output_samples = (output_samples / 2 + 0.5).clamp(0, 1)
    output_samples = output_samples.cpu().float()
    import cv2
    save_folder = output_dir

    for b_id in range(output_samples.size(0)):
        gen_img = np.round(np.clip(output_samples[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(b_id).zfill(5))), gen_img)
    # output_samples = output_samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    # images = []

    # import pdb
    # pdb.set_trace()
    # import matplotlib.pyplot as plt
    # for i, sample in enumerate(output_samples):  
    #     image = Image.fromarray(sample)
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.savefig(f'demo_{i}.png')
    #     images.append(image)

    # return as a pil image
    # image = Image.open(image_path)

    # return images

if __name__ == '__main__':

    main()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(images)
    # plt.savefig('demo.png')
