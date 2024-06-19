# Define paths
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from envs.omnigib.tasks.cube import CubeEnv
from plot_env import fig_to_image, env_to_frames, images_to_video
from envs.omnigib.tasks.cube import CubeEnv

env = CubeEnv()
steps = 3000

path_project = os.path.abspath(os.path.join(__file__, ".."))
path_of_video_with_name = os.path.join(path_project, "videotest.mp4")
print(path_of_video_with_name)


def frames_to_images(frames):
    rewards = [frame['reward'] for frame in frames]
    reward_min, reward_max = min(rewards), max(rewards)

    images = []
    total_steps = len(frames)

    # return images
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the environment frame
        ax[0].imshow(frame['image'])
        ax[0].axis('off')

        # Plot the rewards with fixed axes
        ax[1].plot(rewards[:i + 1])
        ax[1].set_title('Rewards over Steps')
        ax[1].set_xlabel('Step')
        ax[1].set_ylabel('Reward')
        ax[1].set_xlim(0, total_steps)
        if reward_min != reward_max:
            ax[1].set_ylim(reward_min, reward_max)

        # Save the combined image to a buffer
        fig.canvas.draw()
        combined_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        # Convert combined image to BGR (required for cv2.VideoWriter)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGBA2BGR)

        images.append(combined_image)
        plt.close(fig)

    return images


frames = env_to_frames(env, steps)
images = frames_to_images(frames)
images_to_video(images, path_of_video_with_name)
print("Saved video.")
# Close the environment
env.close()
