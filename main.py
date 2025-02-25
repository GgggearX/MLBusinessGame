import os
import torch
import pygame
import matplotlib
from Config import Config
from Logger import Logger
from Train import GomokuEnv, DQNAgent, Trainer
matplotlib.use('Agg')
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark=True

class Plotter:
    @staticmethod
    def plot_curve(data_list,label,title,save_file):
        plt.figure()
        plt.plot(data_list,label=label)
        plt.title(title)
        plt.legend()
        plt.savefig(save_file)
        plt.close()


def play_game(config):
    pygame.init()
    board_size=config.board_size
    cell_size=50
    width=board_size*cell_size
    height=board_size*board_size
    screen=pygame.display.set_mode((board_size*cell_size,board_size*cell_size))
    env=GomokuEnv(board_size,config.win_length,config.agent_player)
    s=env.reset()
    agent=DQNAgent(config)
    if os.path.isfile(config.save_model_file):
        agent.online_net.load_state_dict(torch.load(config.save_model_file,map_location=device))
        agent.online_net.eval()
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        agent.target_net.eval()
        print(f"Loaded model from {config.save_model_file}")
    else:
        print(f"Model not found: {config.save_model_file}, train first.")
        pygame.quit()
        return
    running=True
    clock=pygame.time.Clock()
    while running:
        env.render(screen,cell_size)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
                break
            if event.type==pygame.MOUSEBUTTONDOWN:
                if env.current_player!=config.agent_player and not env.done:
                    pos=pygame.mouse.get_pos()
                    x,y=pos
                    j=x//cell_size
                    i=y//cell_size
                    if 0<=i<board_size and 0<=j<board_size and env.board[i,j]==0 and not env.done:
                        s,r,dn,_=env.step((i,j))
        if env.current_player==config.agent_player and not env.done:
            a_idx=agent.choose_action(s,env)
            i,j=divmod(a_idx,board_size)
            if 0<=i<board_size and 0<=j<board_size and env.board[i,j]==0:
                s,r,dn,_=env.step((i,j))
        if env.done:
            env.render(screen,cell_size)
            font=pygame.font.SysFont('simhei',48)
            if env.winner==config.agent_player:
                text=font.render("AI Wins!",True,(0,0,0))
            elif env.winner is not None:
                text=font.render("Player Wins!",True,(255,0,0))
            else:
                text=font.render("Draw!",True,(0,0,255))
            t_surf=pygame.Surface((text.get_width()+20,text.get_height()+20),pygame.SRCALPHA)
            t_surf.fill((255,255,255,128))
            screen.blit(t_surf,(width//2-t_surf.get_width()//2,height//2-t_surf.get_height()//2))
            text_rect=text.get_rect(center=(width//2,height//2))
            screen.blit(text,text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            running=False
        clock.tick(30)
    pygame.quit()

def main():
    config=Config()
    logger=Logger(config)
    mode=input("1-Train,2-Play:")
    if mode.strip()=="1":
        config.save_to_json()
        trainer=Trainer(config,logger)
        trainer.train()
        Plotter.plot_curve(trainer.win_rate_history,"WinRate","Test Win Rate","win_rate.png")
        print("Train done.")
    else:
        config.load_from_json()
        play_game(config)

if __name__=="__main__":
    main()
