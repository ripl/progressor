import cv2
import imageio


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, render_reward=True):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.render_reward = render_reward
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env, cur_reward=None, total_reward=None, cur_env_reward=None, total_env_reward=None):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.get_pixels_with_width_height(self.render_size, 
                                                         self.render_size)
                
            # Render the current reward/env_reward at the frame and the total reward/env_reward up to the frame
            if self.render_reward:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                font_scale = self.render_size / 512
                font_thickness = max(1, int(font_scale))

                # Define colors
                total_env_color = (255, 255, 0) # Cyan for total environment reward
                cur_env_color = (0, 255, 255) # Yellow for current environment reward
                total_color = (0, 255, 0)       # Green for total reward
                cur_color = (0, 165, 255)     # Orange for current reward

                # Padding for text
                padding = 5

                # Initialize the starting position for the text
                y_pos = self.render_size - padding

                if cur_reward:
                    cur_reward_text = f"cur: {cur_reward:.2f}"
                    cur_pos = (padding, y_pos)
                    cv2.putText(frame, cur_reward_text, cur_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, cur_color, font_thickness, cv2.LINE_AA)
                    text_size, _ = cv2.getTextSize(cur_reward_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    y_pos -= text_size[1] + padding

                if total_reward:
                    total_reward_text = f"total: {total_reward:.2f}"
                    total_pos = (padding, y_pos)
                    cv2.putText(frame, total_reward_text, total_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, total_color, font_thickness, cv2.LINE_AA)
                    text_size, _ = cv2.getTextSize(total_reward_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    y_pos -= text_size[1] + padding

                if cur_env_reward:
                    cur_env_reward_text = f"cur_env: {cur_env_reward:.2f}"
                    cur_env_pos = (padding, y_pos)
                    cv2.putText(frame, cur_env_reward_text, cur_env_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, cur_env_color, font_thickness, cv2.LINE_AA)
                    text_size, _ = cv2.getTextSize(cur_env_reward_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    y_pos -= text_size[1] + padding

                if total_env_reward:
                    total_env_reward_text = f"total_env: {total_env_reward:.2f}"
                    total_env_pos = (padding, y_pos)
                    cv2.putText(frame, total_env_reward_text, total_env_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, total_env_color, font_thickness, cv2.LINE_AA)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            if "adroit" in str(self.save_dir):
                import numpy as np
                self.frames = np.array(self.frames, dtype=np.uint8).transpose(0, 2, 3, 1)
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, render_reward=True):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.render_reward = render_reward
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs, cur_reward=None, total_reward=None, cur_env_reward=None, total_env_reward=None):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            
            # Render the current reward/env_reward at the frame and the total reward/env_reward up to the frame
            if self.render_reward:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                font_scale = self.render_size / 512
                font_thickness = max(1, int(font_scale))

                # Define colors
                total_env_color = (255, 255, 0) # Cyan for total environment reward
                cur_env_color = (0, 255, 255) # Yellow for current environment reward
                total_color = (0, 255, 0)       # Green for total reward
                cur_color = (0, 165, 255)     # Orange for current reward

                # Padding for text
                padding = 5

                # Initialize the starting position for the text
                y_pos = self.render_size - padding

                if cur_reward:
                    cur_reward_text = f"cur: {cur_reward:.2f}"
                    cur_pos = (padding, y_pos)
                    cv2.putText(frame, cur_reward_text, cur_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, cur_color, font_thickness, cv2.LINE_AA)
                    text_size, _ = cv2.getTextSize(cur_reward_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    y_pos -= text_size[1] + padding

                if total_reward:
                    total_reward_text = f"total: {total_reward:.2f}"
                    total_pos = (padding, y_pos)
                    cv2.putText(frame, total_reward_text, total_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, total_color, font_thickness, cv2.LINE_AA)
                    text_size, _ = cv2.getTextSize(total_reward_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    y_pos -= text_size[1] + padding

                if cur_env_reward:
                    cur_env_reward_text = f"cur_env: {cur_env_reward:.2f}"
                    cur_env_pos = (padding, y_pos)
                    cv2.putText(frame, cur_env_reward_text, cur_env_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, cur_env_color, font_thickness, cv2.LINE_AA)
                    text_size, _ = cv2.getTextSize(cur_env_reward_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    y_pos -= text_size[1] + padding

                if total_env_reward:
                    total_env_reward_text = f"total_env: {total_env_reward:.2f}"
                    total_env_pos = (padding, y_pos)
                    cv2.putText(frame, total_env_reward_text, total_env_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, total_env_color, font_thickness, cv2.LINE_AA)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
