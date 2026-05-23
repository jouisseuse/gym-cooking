import dill as pickle
import copy
import numpy as np

from utils.svo import svo_settings


class Bag:
    def __init__(self, arglist, filename):
        self.data = {}
        self.arglist = arglist
        self.directory = "misc/metrics/pickles/"
        self.filename = filename
        self.set_general()

    def set_general(self):
        self.data["level"] = self.arglist.level
        self.data["num_agents"] = self.arglist.num_agents
        self.data["profiling"] = {info : [] for info in ["Delegation", "Navigation", "Total"]}
        self.data["num_completed_subtasks"] = []

        # Checking whether ablation
        if self.arglist.model1 is not None:
            self.data['agent-1'] = self.arglist.model1
        if self.arglist.model2 is not None:
            self.data['agent-2'] = self.arglist.model2
        if self.arglist.model3 is not None:
            self.data['agent-3'] = self.arglist.model3
        if self.arglist.model4 is not None:
            self.data['agent-4'] = self.arglist.model4

        # Record each agent's ground-truth SVO (radians) from the CLI.
        self.data["true_svo"] = {
            "agent-{}".format(i + 1): svo_settings(self.arglist, "agent-{}".format(i + 1))
            for i in range(self.arglist.num_agents)
        }
        self.data["infer_svo"] = bool(getattr(self.arglist, "infer_svo", False))

        # Prepare for agent information
        for info in ["states","actions", "subtasks", "subtask_agents", "bayes", "holding", "incomplete_subtasks"]:
            self.data[info] = {"agent-{}".format(i+1): [] for i in range(self.arglist.num_agents)}
            if info == "bayes":
                self.data[info] = {"agent-{}".format(i+1): {} for i in range(self.arglist.num_agents)}

        # Per-agent SVO traces: posterior over each partner's theta over time.
        # Outer key: inferring agent. Inner key: partner. Value: list of dicts
        # {mean, std, ess, particles, weights} per timestep.
        self.data["svo_posterior"] = {
                "agent-{}".format(i + 1): {} for i in range(self.arglist.num_agents)}
        # Estimate dict each agent currently uses for Level-1 reasoning.
        self.data["partner_svo_estimates"] = {
                "agent-{}".format(i + 1): [] for i in range(self.arglist.num_agents)}


    def set_recipe(self, recipe_subtasks):
        self.data["all_subtasks"] = recipe_subtasks
        self.data["num_total_subtasks"] = len(recipe_subtasks)

    def set_collisions(self, collisions):
        self.data["collisions"] = collisions


    def add_status(self, cur_time, real_agents):
        for a in real_agents:
            self.data["states"][a.name].append(copy.copy(a.location))
            self.data["holding"][a.name].append(a.get_holding())
            self.data["actions"][a.name].append(a.action)
            self.data["subtasks"][a.name].append(a.subtask)
            self.data["subtask_agents"][a.name].append(a.subtask_agent_names)
            self.data["incomplete_subtasks"][a.name].append(a.incomplete_subtasks)

            for task_combo, p in a.delegator.probs.get_list():
                self.data["bayes"][a.name].setdefault(cur_time, [])
                self.data["bayes"][a.name][cur_time].append((task_combo, p))

            # SVO estimate this agent currently uses for its partners.
            self.data["partner_svo_estimates"][a.name].append(
                    dict(getattr(a, "partner_svo_estimates", {})))

            # If the particle filter is running, snapshot each partner's posterior.
            filters = getattr(a.delegator, "svo_filters", None)
            if filters:
                for partner_name, pf in filters.items():
                    self.data["svo_posterior"][a.name].setdefault(partner_name, [])
                    self.data["svo_posterior"][a.name][partner_name].append({
                        "t": cur_time,
                        "mean": pf.posterior_mean(),
                        "std": pf.posterior_std(),
                        "ess": pf.ess(),
                        "particles": np.array(pf.particles, copy=True),
                        "weights":   np.array(pf.weights, copy=True),
                    })

        incomplete_subtasks = set(self.data["all_subtasks"])
        for a in real_agents:
            incomplete_subtasks = incomplete_subtasks & set(a.incomplete_subtasks)
        self.data["num_completed_subtasks"].append(self.data["num_total_subtasks"] - len(incomplete_subtasks))

    def set_termination(self, termination_info, successful):
        self.data["termination"] = termination_info
        self.data["was_successful"] = successful
        for k, v in self.data.items():
            print("{}: {}\n".format(k, v))
        self.data["num_completed_subtasks_end"] = 0 if len(self.data["num_completed_subtasks"]) == 0 else self.data["num_completed_subtasks"][-1]
        print('completed {} / {} subtasks'.format(self.data["num_completed_subtasks_end"], self.data["num_total_subtasks"]))
        pickle.dump(self.data, open(self.directory+self.filename+'.pkl', "wb"))
        print("Saved to {}".format(self.directory+self.filename+'.pkl'))
