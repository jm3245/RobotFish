    def step (self, action):
        self.nStep += 1

        d_prev = float(self._get_obs()[self.observationIndices["distanceTarget"]])

        # Action as acceleration
        action_velocity = self.data.qvel[6:7].copy() + self.actionMultiplier * action * self.dt
        self.do_simulation(action_velocity, self.frame_skip)

        observation = self._get_obs()
        if self.plotHistogram:
            self.obs.append(observation.copy())
            self.act.append(action.copy())

        # Termination condition
        com = self.data.site("COM_0").xpos
        terminated = (not self.noterminate) and bool(float(observation[self.observationIndices["distanceTarget"]]) < 0.05)

        d_new = float(observation[self.observationIndices["distanceTarget"]])
        rewardProgress = d_prev - d_new
        rewardAction = -np.linalg.norm(action)
        terminated = d_new < 0.05
        reward = float(
            20.0 * rewardProgress
            + 0.01 * rewardAction
            + (100.0 if terminated else 0.0)
        )



        self.currentEpisodeReward += reward
        self.prevAction = action.copy()

        ### Logging
        if self.nStep % 100 == 0:
            print(f"\
                Step {self.nStep:,} in {time.time()-self.startTime:.2f}s: \
                |  COM: [{self.data.site('COM_0').xpos[0]:+.2f}, {self.data.site('COM_0').xpos[1]:+.2f}, {self.data.site('COM_0').xpos[2]:+.2f}] \
                |  Target: [{self.target[0]:+.2f}, {self.target[1]:+.2f}] \
                |  Distance: {observation[self.observationIndices['distanceTarget']][0]:.4f} \
                |  ProximityGain: {rewardProgress:.4f} \
                |  Ctrl: [{self.data.ctrl[0]:+.2f}] \
                |  Action: [{action[0]:+.4f}] \
                |  Reward: {reward:.4f}\
                            ")

            if self.plotHistogram:
                nCols = 7
                fig, axs = plt.subplots(5, nCols, figsize=(nCols*2.5, 12))
                fig.subplots_adjust(wspace=0.6, hspace=0.6)
                for i in range(observation.shape[0]):
                    axs[i//nCols, i%nCols].hist(np.array(self.obs)[:, i], bins=50, density=True)
                    axs[i//nCols, i%nCols].set_xlabel(f"Obs {i}")
                    axs[i//nCols, i%nCols].grid()
                for i in range(1, action.shape[0]+1):
                    axs[-1, -i].hist(np.array(self.act)[:, i-1], bins=50, density=True)
                    axs[-1, -i].set_xlabel(f"Act {i-1}")
                    axs[-1, -i].grid()
                fig.savefig("Outputs/testo.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

        info = {}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

