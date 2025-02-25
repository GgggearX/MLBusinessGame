import os
import json


class Config:
    def __init__(self):
        self.board_size=9
        self.win_length=5
        self.agent_player=1
        self.episodes=5000
        self.test_interval=500
        self.test_games=50
        self.learning_rate=1e-4
        self.gamma=0.95
        self.n_step=3
        self.multi_step=True
        self.capacity=200000
        self.batch_size=256
        self.update_target_freq=1000
        self.use_per=True
        self.per_alpha=0.6
        self.per_beta_start=0.4
        self.per_beta_frames=200000
        self.noisy_net=True
        self.eps_start=1.0
        self.eps_min=0.01
        self.eps_decay=1.0
        self.use_distributional=True
        self.atoms=51
        self.v_min=-10
        self.v_max=10
        self.save_model_file="fast_dqn_model.pth"
        self.save_figure_file="fast_training_curve.png"
        self.save_config_file="fast_config.json"
        self.logs_dir="logs_fast"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def save_to_json(self,filename=None):
        if filename is None:
            filename=self.save_config_file
        data=self.__dict__.copy()
        with open(filename,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=4)
        print(f"[Config] Saved to {filename}")

    def load_from_json(self,filename=None):
        if filename is None:
            filename=self.save_config_file
        if not os.path.isfile(filename):
            print(f"[Config] {filename} not found, skip load")
            return
        with open(filename,"r",encoding="utf-8") as f:
            data=json.load(f)
            for k,v in data.items():
                setattr(self,k,v)
        print(f"[Config] Loaded from {filename}")