import os
import gym
import numpy as np
import cv2
from tqdm.auto import tqdm

import matplotlib.pyplot as plt


def env_to_frames(env, steps):
    frames = []
    # Simulate the environment
    for _ in tqdm(range(steps)):
        image = env.render(width=64, height=64)
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

        frame = {
            'image': image,
            'state': state,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info,
            'action': action
        }
        frames.append(frame)
        if terminated or truncated:
            env.reset()

    return frames


def fig_to_image(fig):
    # Save the combined image to a buffer
    fig.canvas.draw()
    combined_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # Convert combined image to BGR (required for cv2.VideoWriter)
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGBA2BGR)
    return combined_image


def images_to_video(images, video_path):

    # Check that all variables have the same length
    # total_steps = len(images)
    # # Create combined images with Matplotlib
    # combined_frames = []
    # for i, frame in tqdm(enumerate(image_frames), total=len(image_frames)):
    #     fig, ax = plt.subplots(1, len(kwargs) + 1, figsize=(6 * (len(kwargs) + 1), 6))
    #
    #     # Plot the environment frame
    #     ax[0].imshow(frame)
    #     ax[0].axis('off')
    #
    #     # Plot the additional variables
    #     for j, (var_name, var_list) in enumerate(kwargs.items()):
    #         ax[j + 1].plot(var_list[:i + 1], color='blue')
    #         ax[j + 1].set_title(var_name)
    #         ax[j + 1].set_xlabel('Step')
    #         ax[j + 1].set_ylabel(var_name)
    #         ax[j + 1].set_xlim(0, total_steps)
    #         ax[j + 1].set_ylim(limits[var_name][0], limits[var_name][1])
    #
    #     combined_frames.append(combined_image)
    #     plt.close(fig)

    # Define video writer
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    # Write combined frames to video
    for frame in images:
        video.write(frame)

    # Release the video writer
    video.release()
