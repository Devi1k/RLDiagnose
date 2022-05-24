from collections import deque

import numpy as np


class PrioritizedReplayBuffer(object):

    def __init__(self, buffer_size):
        self._priorities = deque(maxlen=buffer_size)


    def __len__(self):
        return len(self._priorities)

    def add(self, state, action, reward, next_state, episode_over, error):
        self._priorities.append((state, action, reward, next_state, episode_over, error ))

    def sample(self, batch_size, priority_scale=1.0):
        batch_size = min(len(self._priorities), batch_size)
        batch_probs = self.get_probabilities(priority_scale)
        #print(len(self._priorities),len(batch_probs))
        #print(batch_probs)
        batch_indices = np.random.choice(range(len(self._priorities)), size=batch_size, p=batch_probs)
        #batch_importance = self.get_importance(batch_probs[batch_indices])
        batch = [self._priorities[x][:5] for x in batch_indices]

        return batch

    def get_probabilities(self, priority_scale):
        td_errors = np.array([abs(x[5]) for x in self._priorities])
        #print(td_errors)
        scaled_priorities = td_errors ** priority_scale
        batch_probabilities = scaled_priorities / sum(scaled_priorities)
        return batch_probabilities

    def get_importance(self, probabilities):
        importance = 1 / (len(self._priorities) * probabilities+0.001)
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self._priorities[i] = abs(e) + offset