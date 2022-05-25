def eval_agent(env, agent, times=1, print_score=False, render=False):
    scores = []

    for _ in range(times):
        obs = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(obs, train_mode=False)
            next_obs, reward, done, _ = env.step(action)
            if render:
                env.render()

            obs = next_obs
            score += reward

        scores.append(score)
        if print_score:
            print(score)

    return sum(scores) / len(scores)


def test(env, agent):
    agent.load()

    score = eval_agent(env, agent, print_score=True, times=50)
    print(f"avt score: {score:.2f}")


def try_checkpoint(env, agent, best_eval_score):
    current_eval_score = eval_agent(env, agent, times=20)
    if current_eval_score > best_eval_score:
        best_eval_score = current_eval_score
        print(f"checkpoint eval_score={current_eval_score:.2f}")
        agent.save()

    return best_eval_score
