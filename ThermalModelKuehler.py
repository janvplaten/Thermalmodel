import math
import numpy as np
import torch
import torch.nn as nn
import os
from torch.distributions import Normal
import matplotlib.pyplot as plt


# ============================================================
# 1. Grundeinstellungen
# ============================================================

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Verwendetes Device:", DEVICE)

# ============================================================
# 2. Thermisches Model
# ============================================================

class ThermalModel:

    def __init__(self, dt):

        self.dt = dt

        T_Air   = np.random.uniform(10.0, 50.0)
        T_Cool  = np.random.uniform(10.0, 50.0)
        T_Batt  = np.random.uniform(10.0, 50.0)

        self.x = np.array([
            T_Air,
            T_Cool,
            T_Cool,
            T_Cool,
            T_Batt
            ], dtype=np.float64)

        self.cp_KW = 4200

        self.cp = np.array([
            1,
            self.cp_KW * 0.5,
            self.cp_KW * 0.5,
            self.cp_KW * 0.5,
            6000
        ])

        self.rho = 1000

        # Wärmeübergangskoeffizienten
        self.alpha_KW_HX = np.array([
            1.0, 1000.0, 2000.0, 3000.0, 4000.0
        ])

        self.alpha_KW_Batt = np.array([
            1.0, 1000.0, 2000.0, 3000.0, 4000.0
        ])

        self.BP_Vp_alpha_Vp = np.array([
            0.0, 5.0, 10.0, 15.0, 20.0
        ])

        self.alpha_Air_Hx = np.array([
            10.0, 100.0, 1000.0, 2000.0, 2500.0, 3000.0, 3250.0
        ])

        self.BP_alpha_Air_Hx = np.array([
            0.0, 10.0, 50.0, 100.0, 150.0, 200.0, 250.0
        ])

        # Grenzen der Stellgrößen        
        self.Vp_min = 0.1
        self.Vp_max = 20.0
        self.V_max = 200.0
        self.V_add_max = 50.0
        self.Qh_max = 1000.0
        self.Bypass = 1.0

        self.flows = {k: 0.0 for k in
                ["Q_Hx", "Q_Heater", "Q_Batt", "P_Batt",
                "Q_Mix_1", "Q_Mix_2", "Q_Mix_3"]}

    def reset(self):

        T_Air = np.random.uniform(20.0, 30.0)
        T_Cool = np.random.uniform(25.0, 35.0)
        T_Bat = np.random.uniform(25.0, 35.0)
            
        self.x[0] = T_Air
        self.x[1] = T_Cool
        self.x[2] = T_Cool
        self.x[3] = T_Cool
        self.x[4] = T_Bat

    def set(self,T_Air, T_Cool, T_Bat):
            
        self.x[0] = T_Air
        self.x[1] = T_Cool
        self.x[2] = T_Cool
        self.x[3] = T_Cool
        self.x[4] = T_Bat

    def Q_tans_Hx(self, Vp, V, V_add):

        al_Hx_Cool  = np.interp( Vp, self.BP_Vp_alpha_Vp, self.alpha_KW_HX)
        al_Hx_Air   = np.interp( V + V_add, self.BP_alpha_Air_Hx, self.alpha_Air_Hx)

        al_tans = 1 / (1 / al_Hx_Air + 1 / al_Hx_Cool)
        Q = al_tans * (self.x[0] - self.x[1])
        return float(Q)

    def Q_tans_Batt(self, Vp):

        al_Hx_Cool  = np.interp( Vp, self.BP_Vp_alpha_Vp, self.alpha_KW_Batt)

        Q = al_Hx_Cool * (self.x[3] - self.x[4])
        return float(Q)

    def Q_Mix(self, Vp, T_in, T_out, Bypass):

        mp = self.rho * Vp * 0.001 / 60.0 * (1 - Bypass)

        Q = (T_in - T_out) * self.cp_KW * mp

        return float(Q)
    
    def step(self, Qh, Vp, V, V_add, Bypass):


        Q_Hx    = self.Q_tans_Hx(Vp, V, V_add)
        Q_Batt  = self.Q_tans_Batt(Vp)
        P_bat = ((V / 200) * 31) ** 2

        Q_trans = np.array([
            0,
            Q_Hx,
            Qh,
            -Q_Batt,
            Q_Batt + P_bat
        ])

        Q_mix = np.array([
            0,
            self.Q_Mix(Vp, self.x[3], self.x[1], Bypass),
            self.Q_Mix(Vp, self.x[1] * (1 - Bypass) + self.x[3] * Bypass, self.x[2], 0),
            self.Q_Mix(Vp, self.x[2], self.x[3], 0),
            0
        ])


        dT = (Q_trans + Q_mix) * self.dt / self.cp

        # Wärmeströme für Auswertung/Plot ablegen (Werte vor dem Zeitschritt)
        self.flows = {
            "Q_Hx":     Q_Hx,            # Luft -> Kühlwasser (Wärmetauscher)
            "Q_Heater": Qh,              # Heizer -> Kühlwasser
            "Q_Batt":   Q_Batt,          # Kühlwasser -> Batterie
            "P_Batt":   P_bat,           # Verlustleistung der Batterie
            "Q_Mix_1":  Q_mix[1],        # Mischströme der Kühlwasserknoten
            "Q_Mix_2":  Q_mix[2],
            "Q_Mix_3":  Q_mix[3],
        }

        self.x = self.x + dT

class EnvThermalNetwerk:

    # Inputs
    # T Air
    # T Cool 1
    # T Cool 2
    # T Cool 3
    # T Batt Ist
    # T Batt Soll
    # P Batt
    # V Vehical


    def __init__(self, episode_len):

        self.sys = ThermalModel(0.2)

        self.OBS_Mean    = np.array([20.0, 35.0, 35.0, 35.0, 30.0, 30.0, 1000.0, 100.0, 0, 0, 0, 0])
        self.OBS_Std     = np.array([30.0, 45.0, 45.0, 45.0, 40.0, 40.0, 1000.0, 100.0, 1, 1, 1, 1])

        self.episode_len = episode_len

        self.action_dim = 4
        self.obs_dim = 12

        self.last_action = np.zeros(
            self.action_dim,
            dtype=np.float32
        )

        self.t = 0

    def select_Cycle(self, Input, Vmax):

        n_steps = int(self.episode_len / self.sys.dt) 

        self.Cycle_t    = torch.arange(n_steps + 1, dtype=torch.float32, device=DEVICE) * self.sys.dt
        T       = 60
        match Input:
            case "sin":
                self.Cycle_V = (torch.sin(2 * math.pi * self.Cycle_t / T) + 1 ) * Vmax/2
            case "const":
                self.Cycle_V = torch.full((n_steps+1,), Vmax, dtype=torch.float32, device=DEVICE)


    def reset(self):

        self.T_set = np.random.uniform(33.0, 40.0)

        self.select_Cycle("sin", 200)

        self.sys.reset()

        if self.T_set < self.sys.x[0] + 5:
            self.T_set = self.sys.x[0] + 5

        self.last_action = np.zeros(
            self.action_dim,
            dtype=np.float32
        )

        self.t = 0

        return self._obs()

    def set(self, T_Air, T_Cool, T_Bat, V, T_set):

        self.T_set = T_set

        self.select_Cycle("sin", V)

        self.sys.set(T_Air, T_Cool, T_Bat)

        if self.T_set < self.sys.x[0] + 5:
            self.T_set = self.sys.x[0] + 5

        self.last_action = np.zeros(
            self.action_dim,
            dtype=np.float32
        )

        self.t = 0

        return self._obs()

    def _obs(self):

        V = np.interp( self.t, self.Cycle_t, self.Cycle_V)

        P_bat = ((V / 200) * 31) ** 2

        obs = np.array([
            self.sys.x[0],
            self.sys.x[1],
            self.sys.x[2],
            self.sys.x[3],
            self.sys.x[4],
            self.T_set,
            P_bat,
            V,
            self.last_action[0],
            self.last_action[1],
            self.last_action[2],
            self.last_action[3]
        ])

        return (obs - self.OBS_Mean) / self.OBS_Std

    def step(self,action):
        Vp = self.sys.Vp_min + (action[0] + 1) * 0.5 * (self.sys.Vp_max - self.sys.Vp_min)
        V_add   = (action[1] + 1) * 0.5 * self.sys.V_add_max
        Qh      = (action[2] + 1) * 0.5 * self.sys.Qh_max
        Bypass  = (action[3] + 1) * 0.5 * self.sys.Bypass

        V = np.interp( self.t, self.Cycle_t, self.Cycle_V)

        self.sys.step(Qh, Vp, V, V_add, Bypass)

        self.t += self.sys.dt

        Tb = self.sys.x[4]

        # Sicherheitsgrenzen
        unsafe = Tb > 80.0 or Tb < 0.0

        # Episodenende
        done = (self.t >= self.episode_len or unsafe )

        err = Tb - self.T_set

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        action_change = float(np.sum(np.abs(action - self.last_action)))

        flow_cost = (Vp / self.sys.Vp_max)

        fan_cost = (V_add / self.sys.V_add_max)

        heater_cost = (Qh / self.sys.Qh_max)

        simultaneity_cost = fan_cost * heater_cost

        reward = 0
        reward -= 0.05 * err ** 2
        reward -= 0.05 * heater_cost
        reward -= 0.001 * fan_cost
        reward -= 1e-5 * flow_cost

        if err+2 > 0:
            reward -= 10 * heater_cost ** 2

        reward -= 0.01 * Bypass * err
        reward += 0.01 * fan_cost * err

        if unsafe:
            reward -= 100.0

        simultaneity_penalty = (20.0 * max(0.0, simultaneity_cost - 0.05))

        reward -= simultaneity_penalty

        self.last_action = action.copy()


        info = {
            "Tb": Tb,
            "T_set": self.T_set,
            "err": abs(err),
            "Vp": Vp,
            "V_add": V_add,
            "Qh": Qh,
        }
        
        return (self._obs(), float(reward), done, info)
    
# ============================================================
# 4. Actor-Critic Netzwerk
# ============================================================

class ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.actor = nn.Linear(32, act_dim)
        self.critic = nn.Linear(32, 1)
        self.actor_logstd = nn.Parameter(torch.full((act_dim,), -0.5))
        self.apply(self.init_weights)
        nn.init.orthogonal_(self.actor.weight,gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight,gain=1.0)
        nn.init.zeros_(self.critic.bias)

    @staticmethod
    def init_weights(module):

        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight,gain=math.sqrt(2))
            nn.init.zeros_(module.bias)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, obs):
        features = self.body(obs)
        mu = self.actor(features)
        value = self.critic(features).squeeze(-1)
        logstd = torch.clamp(self.actor_logstd,-5.0,0.5)
        std = torch.exp(logstd).expand_as(mu)
        return mu, std, value

    # --------------------------------------------------------
    # Log-Wahrscheinlichkeit mit Tanh-Korrektur
    # --------------------------------------------------------

    def get_action_logprob(self,mu,std,raw_action):
        dist = Normal(mu, std)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Numerisch stabile Tanh-Jacobi-Korrektur
        correction = 2.0 * (math.log(2.0) - raw_action - nn.functional.softplus(-2.0 * raw_action))
        log_prob = log_prob - correction.sum(dim=-1)
        return log_prob

    # --------------------------------------------------------
    # Aktion auswählen
    # --------------------------------------------------------

    def act(self, obs):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        log_prob = self.get_action_logprob(mu,std,raw_action)
        return (action,raw_action,log_prob,value)

    # --------------------------------------------------------
    # Policy auswerten
    # --------------------------------------------------------

    def evaluate(self, obs, raw_actions):

        mu, std, value = self.forward(obs)

        log_prob = self.get_action_logprob(mu,std,raw_actions)

        entropy = Normal(mu,std).entropy().sum(dim=-1)

        return log_prob, entropy, value


# ============================================================
# 5. PPO Algorithmus
# ============================================================

class PPO:

    def __init__(self, env, lr=3e-4, gamma=0.99, gae_lambda=0.95,
             clip=0.20, value_clip=5.0, epochs=10, batch_size=256,
             rollout=8192, entropy_coef=0.0, value_coef=0.5):

        self.env = env

        self.net = ActorCritic(
            env.obs_dim,
            env.action_dim
        ).to(DEVICE)

        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
            eps=1e-5
        )

        self.gamma = gamma
        self.lam = gae_lambda

        self.clip = clip
        self.value_clip = value_clip

        self.epochs = epochs
        self.batch = batch_size
        self.rollout = rollout

        self.ent_coef = entropy_coef
        self.val_coef = value_coef

        self.obs = self.env.reset()

        self.current_ep_reward = 0.0
        self.reward_history = []
        self.episode_history = []
        self.temperature_history = []
        self.clip_history = []

    # --------------------------------------------------------
    # Rollout sammeln
    # --------------------------------------------------------

    def collect_rollout(self):

        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []
        ep_rewards = []


        self.net.eval()

        for step in range(self.rollout):

            obs_tensor = torch.as_tensor(self.obs,dtype=torch.float32,device=DEVICE)

            with torch.no_grad():
                action, raw_action, logp, value = self.net.act(obs_tensor)

            action_np = action.cpu().numpy()

            next_obs, reward, done, info = self.env.step(action_np)

            # Daten speichern
            obs_buf.append(obs_tensor)
            act_buf.append(raw_action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(float(done))

            self.current_ep_reward += reward

            if done:
                ep_rewards.append(self.current_ep_reward)
                self.current_ep_reward = 0.0
                self.obs = self.env.reset()

            else:
                self.obs = next_obs

            
            # Temperatur für Monitoring
            self.temperature_history.append(info["Tb"])


        # Wert des letzten Zustands
        obs_tensor = torch.as_tensor(self.obs,dtype=torch.float32,device=DEVICE)

        with torch.no_grad():
            _, _, last_val = self.net(obs_tensor)

        # Tensoren erstellen
        obs_buf = torch.stack(obs_buf)
        act_buf = torch.stack(act_buf)
        logp_buf = torch.stack(logp_buf)
        val_buf = torch.stack(val_buf)
        rew_buf = torch.tensor(rew_buf,dtype=torch.float32,device=DEVICE)

        done_buf = torch.tensor(done_buf,dtype=torch.float32,device=DEVICE)

        avg_reward = rew_buf.mean().item()

        return (obs_buf,act_buf,logp_buf,rew_buf,val_buf,done_buf,last_val,avg_reward,ep_rewards)

    # --------------------------------------------------------
    # GAE
    # --------------------------------------------------------

    def compute_gae(self,rewards,values,dones,last_val):

        T = len(rewards)

        advantages = torch.zeros_like(rewards)

        gae = torch.zeros((),dtype=torch.float32,device=DEVICE)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_val
            else:
                next_value = values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = (rewards[t] + self.gamma * next_value * non_terminal - values[t])
            gae = ( delta + self.gamma * self.lam * non_terminal * gae)
            advantages[t] = gae

        returns = advantages + values

        # Advantage normalisieren
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return advantages, returns

    # --------------------------------------------------------
    # PPO Update
    # --------------------------------------------------------

    
    def update(self, data):

        obs, acts, old_logp, old_values, advantages, returns = data

        n = len(obs)

        clipped_count = 0
        total_count = 0

        self.net.train()

        for epoch in range(self.epochs):

            indices = torch.randperm(n, device=DEVICE)

            for start in range(0, n, self.batch):

                batch_indices = indices[start:start + self.batch]

                # Policy und Critic auswerten
                logp, entropy, new_values = self.net.evaluate(
                    obs[batch_indices],
                    acts[batch_indices]
                )

                # ------------------------------------------------
                # Policy Loss (PPO Clipping)
                # ------------------------------------------------

                log_ratio = logp - old_logp[batch_indices]
                ratio = torch.exp(log_ratio)

                lower_bound = 1.0 - self.clip
                upper_bound = 1.0 + self.clip

                ratio_was_clipped = (
                    (ratio < lower_bound) |
                    (ratio > upper_bound)
                )

                clipped_count += int(ratio_was_clipped.sum().item())
                total_count += len(ratio_was_clipped)

                surr1 = ratio * advantages[batch_indices]

                surr2 = torch.clamp(
                    ratio,
                    lower_bound,
                    upper_bound
                ) * advantages[batch_indices]

                policy_loss = -torch.min(surr1, surr2).mean()

                # ------------------------------------------------
                # Value Clipping
                # ------------------------------------------------

                old_values_batch = old_values[batch_indices]

                returns_batch = returns[batch_indices]

                # Geclippte Value-Vorhersage
                clipped_values = old_values_batch + torch.clamp(
                    new_values - old_values_batch,
                    -self.value_clip,
                    self.value_clip
                )

                # Normaler Critic Loss
                value_loss_unclipped = (
                    new_values - returns_batch
                ).pow(2)

                # Geclippter Critic Loss
                value_loss_clipped = (
                    clipped_values - returns_batch
                ).pow(2)

                # Maximum beider Verluste
                value_loss = 0.5 * torch.maximum(
                    value_loss_unclipped,
                    value_loss_clipped
                ).mean()

                # ------------------------------------------------
                # Entropie
                # ------------------------------------------------

                entropy_loss = entropy.mean()

                # ------------------------------------------------
                # Gesamtverlust
                # ------------------------------------------------

                loss = (
                    policy_loss
                    + self.val_coef * value_loss
                    - self.ent_coef * entropy_loss
                )

                self.opt.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.net.parameters(),
                    max_norm=0.5
                )

                self.opt.step()

        clipped_percentage = (
            100.0 * clipped_count / total_count
            if total_count > 0 else 0.0
        )

        return clipped_percentage


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train(self, iterations=300):

        for it in range(1, iterations + 1):

            # -----------------------------------------------
            # Entropy Annealing
            # -----------------------------------------------

            progress = (it - 1) / max(iterations - 1, 1)

            data = self.collect_rollout()

            (obs, acts, logp, rewards, values,
            dones, last_val, avg_reward, ep_rewards) = data

            advantages, returns = self.compute_gae(
                rewards, values, dones, last_val
            )

            clipped_percentage = self.update(
                (obs, acts, logp, values, advantages, returns)
            )

            self.clip_history.append(clipped_percentage)
            self.reward_history.append(avg_reward)

            if len(ep_rewards) > 0:
                mean_ep_reward = np.mean(ep_rewards)
                self.episode_history.append(mean_ep_reward)
            else:
                mean_ep_reward = np.nan

            if it % 10 == 0:

                std = torch.exp(
                    torch.clamp(
                        self.net.actor_logstd.detach(),
                        -5.0,
                        0.5
                    )
                )

                print(
                    f"Iteration {it:4d} | "
                    f"Reward/Step: {avg_reward:9.3f} | "
                    f"Episode Reward: {mean_ep_reward:10.2f} | "
                    f"Std: {std.mean().item():.4f} | "
                    f"LogStd: {self.net.actor_logstd.mean().item():.4f}"
                )

        return self.net
# ============================================================
# 6. Training
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Umgebung
    # --------------------------------------------------------

    env = EnvThermalNetwerk(episode_len=600)

    # --------------------------------------------------------
    # PPO Agent
    # --------------------------------------------------------

    agent = PPO(env)

    # --------------------------------------------------------
    # Training starten
    # --------------------------------------------------------

    net = agent.train(iterations=500)

    # Modell speichern
    save_path = os.path.expanduser("~/battery_thermal_ppo.pt")

    torch.save(net.state_dict(),save_path)

    print("Modell gespeichert unter:", save_path)
    print("\nTraining abgeschlossen.")


    # ========================================================
    # 7. Trainingsverlauf
    # ========================================================

    fig, ax = plt.subplots(2, 1,figsize=(11, 7))

    ax[0].plot(agent.reward_history)
    ax[0].set_title("Mittlerer Reward pro Rollout")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Reward / Step")
    ax[0].grid(True)
    ax[1].plot(agent.episode_history)
    ax[1].set_title("Reward pro abgeschlossener Episode")
    ax[1].set_xlabel("Abgeschlossene Episode")
    ax[1].set_ylabel("Episode Reward")
    ax[1].grid(True)
    fig.tight_layout()
    plt.show(block=False)

    # ========================================================
    # 8. Simulation mit trainiertem Regler
    # ========================================================

    env = EnvThermalNetwerk(episode_len=3600)

    obs = env.reset()

    n_steps = int(env.episode_len / env.sys.dt)

    T_Air       = np.zeros(n_steps + 1)
    T_KW_2_0    = np.zeros(n_steps + 1)
    T_KW_2_1    = np.zeros(n_steps + 1)
    T_KW_2_2    = np.zeros(n_steps + 1)
    T_Batt_0    = np.zeros(n_steps + 1)
    Vp          = np.zeros(n_steps)
    V_add       = np.zeros(n_steps)
    P_Heiz      = np.zeros(n_steps)
    Bypass      = np.zeros(n_steps)
    Q_Hx        = np.zeros(n_steps)
    Q_Heater    = np.zeros(n_steps)
    Q_Batt      = np.zeros(n_steps)
    P_Batt      = np.zeros(n_steps)
    Q_Mix_1     = np.zeros(n_steps)
    Q_Mix_2     = np.zeros(n_steps)
    Q_Mix_3     = np.zeros(n_steps)

    # Solltemperatur speichern
    T_set = env.T_set

    # Anfangswerte
    T_Air[0]    = env.sys.x[0]
    T_KW_2_0[0] = env.sys.x[1]
    T_KW_2_1[0] = env.sys.x[2]
    T_KW_2_2[0] = env.sys.x[3]
    T_Batt_0[0] = env.sys.x[4]


    # Deterministische Policy
    net.eval()

    for i in range(n_steps):

        obs_tensor = torch.as_tensor(obs,dtype=torch.float32,device=DEVICE)

        with torch.no_grad():
            mu, std, value = net(obs_tensor)
            action = torch.tanh(mu)

        action_np = action.cpu().numpy()
        next_obs, reward, done, info = env.step(action_np)

        f = env.sys.flows
        Q_Hx[i]     = f["Q_Hx"]
        Q_Heater[i] = f["Q_Heater"]
        Q_Batt[i]   = f["Q_Batt"]
        P_Batt[i]   = f["P_Batt"]
        Q_Mix_1[i]  = f["Q_Mix_1"]
        Q_Mix_2[i]  = f["Q_Mix_2"]
        Q_Mix_3[i]  = f["Q_Mix_3"]

        # Physikalische Stellgrößen
        Vp[i] = env.sys.Vp_min + (action_np[0] + 1.0) * 0.5 * (env.sys.Vp_max - env.sys.Vp_min)
        V_add[i]    = ((action_np[1] + 1.0) * 0.5 * env.sys.V_add_max)
        P_Heiz[i]   = ((action_np[2] + 1.0) * 0.5 * env.sys.Qh_max)
        Bypass[i]   = ((action_np[3] + 1.0) * 0.5 * env.sys.Bypass)

        # Temperaturen speichern
        T_Air[i+1]      = env.sys.x[0]
        T_KW_2_0[i+1]   = env.sys.x[1]
        T_KW_2_1[i+1]   = env.sys.x[2]
        T_KW_2_2[i+1]   = env.sys.x[3]
        T_Batt_0[i+1]   = env.sys.x[4]

        # Beobachtung aktualisieren
        obs = next_obs

        if done:
            print(f"Simulation vorzeitig beendet bei t={i+1} s")
            break

    # ========================================================
    # 9. Simulation plotten
    # ========================================================

    t = np.arange(n_steps + 1) * env.sys.dt
    t_steps = t[1:]
    fig, (ax_temp, ax1, ax_Q) = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # --------------------------------------------------------
    # Temperaturen
    # --------------------------------------------------------

    ax_temp.plot(t,T_Air, label="T Air")
    ax_temp.plot(t,T_KW_2_0,label="KW 2.0")
    ax_temp.plot(t,T_KW_2_1,label="KW 2.1")
    ax_temp.plot(t,T_KW_2_2,label="KW 2.2")
    ax_temp.plot(t,T_Batt_0,label="Batterie",linewidth=2)
    ax_temp.axhline(y=T_set,color="black",linestyle="--",label="Solltemperatur")
    ax_temp.axhline(y=80.0,color="red",linestyle=":",label="Max. Temperatur")
    ax_temp.set_ylabel("Temperatur [°C]")
    ax_temp.set_title("Temperaturverlauf des thermischen Modells")
    ax_temp.legend(loc="best")
    ax_temp.grid(True)

    # --------------------------------------------------------
    # Volumenströme
    # --------------------------------------------------------

    ax1.plot(t_steps,Vp,label="Vp")
    ax1.plot(t_steps,V_add,label="V add")
    ax1.set_ylabel("Volumenstrom [l/min]")
    ax1.set_ylim(0, 52)
    ax1.grid(True)

    # --------------------------------------------------------
    # Heizleistung
    # --------------------------------------------------------

    ax2 = ax1.twinx()
    ax2.plot( t_steps, P_Heiz, color="tab:red",label="P Heizer")
    ax2.plot( t_steps, Bypass * env.sys.Qh_max, color="black",label="Bypass")
    ax2.set_ylabel("Heizleistung [W]")
    ax2.set_ylim(0, 1020)

    # Gemeinsame Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2,labels1 + labels2,loc="upper right")
    ax1.set_xlabel("Zeit [s]")
    ax1.set_title("Stellgrößen des trainierten PPO-Reglers")
    fig.tight_layout()

    # --------------------------------------------------------
    # Wärmeströme
    # --------------------------------------------------------

    ax_Q.plot(t_steps, Q_Hx,     label="Q Wärmetauscher (Luft → KW)")
    ax_Q.plot(t_steps, Q_Heater, label="Q Heizer (→ KW)")
    ax_Q.plot(t_steps, Q_Batt,   label="Q KW → Batterie")
    ax_Q.plot(t_steps, P_Batt,   label="P Batterie (Verlustleistung)")
    # ax_Q.plot(t_steps, Q_Mix_1,  linestyle="--", label="Q Mix KW 2.0")
    # ax_Q.plot(t_steps, Q_Mix_2,  linestyle="--", label="Q Mix KW 2.1")
    # ax_Q.plot(t_steps, Q_Mix_3,  linestyle="--", label="Q Mix KW 2.2")
    ax_Q.axhline(0, color="black", linewidth=0.8)
    ax_Q.set_ylabel("Wärmestrom [W]")
    ax_Q.set_xlabel("Zeit [s]")
    ax_Q.set_title("Wärmeströme im thermischen Modell")
    ax_Q.legend(loc="upper right", ncol=2, fontsize=8)
    ax_Q.grid(True)



    # ========================================================
    # 10. Ergebnisse
    # ========================================================

    print("\nSimulationsergebnisse")
    print("Endtemperatur Batterie:",round(T_Batt_0[min(i+1, n_steps)], 2),"°C")
    print("Solltemperatur:",T_set,"°C")
    print("Temperaturabweichung:",round(abs(T_Batt_0[min(i+1, n_steps)] - T_set),2),"K")
    print("Mittlerer Volumenstrom:",round(np.mean(Vp), 2),"l/min")
    print("Mittlerer Lüfter Geschwindigkeit:",round(np.mean(V_add), 2),"l/min")
    print("Mittlere Heizleistung:",round(np.mean(P_Heiz), 2),"W")
    plt.show()