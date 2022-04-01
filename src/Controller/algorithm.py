import agents.controller as controller
import agents.dqn as dqn
import agents.ppo as ppo
import agents.a2c as a2c
import agents.a2cmix as a2cmix
import agents.coma as coma
import agents.maddpg as maddpg
import agents.ppomix as ppomix
import agents.vdn as vdn
import agents.qmix as qmix
import agents.UPDeT as updet
import agents.ppo_continuous as ppo_con
import argparse

def make(algorithm, params={}):
    if algorithm == "Random":
        params["adversary_ratio"] = 0
        return controller.Controller(params)
    if algorithm == "DQN":
        params["adversary_ratio"] = 0
        return dqn.DQNLearner(params)
    if algorithm == "PPO":
        params["adversary_ratio"] = 0
        return ppo.PPOLearner(params)
    if algorithm == "PPOC":
        params["adversary_ratio"] = 0
        return ppo_con.PPO_con_Learner(params)
    if algorithm == "IAC":
        params["adversary_ratio"] = 0
        return a2c.A2CLearner(params)
    if algorithm == "RAT_IAC":
        params["adversary_ratio"] = None
        return a2c.A2CLearner(params)
    if algorithm == "AC-QMIX":
        params["adversary_ratio"] = 0
        params["central_q_learner"] = qmix.QMIXLearner(params)
        return a2cmix.A2CMIXLearner(params)
    if algorithm == "RADAR_X":
        assert params["adversary_ratio"] is not None, "RADAR (X), requires adversary-ratio as float"
        params["central_q_learner"] = vdn.VDNLearner(params)
        return a2cmix.A2CMIXLearner(params)
    if algorithm == "RADAR":
        params["adversary_ratio"] = None
        params["central_q_learner"] = vdn.VDNLearner(params)
        return a2cmix.A2CMIXLearner(params)
    if algorithm == "COMA":
        params["adversary_ratio"] = 0
        return coma.COMALearner(params)
    if algorithm == "MADDPG":
        params["minimax"] = False
        assert params["adversary_ratio"] is not None, "MADDPG, requires adversary-ratio as float"
        return maddpg.MADDPGLearner(params)
    if algorithm == "M3DDPG":
        params["minimax"] = True
        params["adversary_ratio"] = 0 # Adversaries are modeled within Q-function
        return maddpg.MADDPGLearner(params)
    if algorithm == "PPO-QMIX":
        params["adversary_ratio"] = 0
        params["central_q_learner"] = qmix.QMIXLearner(params)
        return ppomix.PPOMIXLearner(params)
    if algorithm == "RADAR_PPO":
        params["adversary_ratio"] = None
        params["central_q_learner"] = vdn.VDNLearner(params)
        return ppomix.PPOMIXLearner(params)
    if algorithm == "RAT_PPO":
        params["adversary_ratio"] = None
        return ppo.PPOLearner(params)
    if algorithm == "RAT_DQN":
        params["adversary_ratio"] = None
        return dqn.DQNLearner(params)
    if algorithm == "VDN":
        params["adversary_ratio"] = 0
        return vdn.VDNLearner(params)
    if algorithm == "QMIX":
        params["adversary_ratio"] = 0
        return qmix.QMIXLearner(params)
    if algorithm == "UPDeT-QMIX":
        params["adversary_ratio"] = 0
        parser = argparse.ArgumentParser(description='Unit Testing')
        parser.add_argument('--token_dim', default='5', type=int)
        parser.add_argument('--emb', default='32', type=int)
        parser.add_argument('--heads', default='3', type=int)
        parser.add_argument('--depth', default='2', type=int)
        parser.add_argument('--ally_num', default='5', type=int)
        parser.add_argument('--enemy_num', default='5', type=int)
        parser.add_argument('--episode', default='20', type=int)
        args = parser.parse_args()
        return updet.UPDeT(None, args)
    raise ValueError("Unknown algorithm '{}'".format(algorithm))