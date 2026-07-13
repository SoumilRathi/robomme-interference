# Retrieval scripts

Two scripts for the retrieval step described in the "Adding retrieval" section. They run on dumped episode histories and give, for each episode, the frames to hand the policy.

An episode is a directory with `frames.npz`, which holds the history frames and the query frame, and `manifest.json`. We work from the frames. We read the session labels in the manifest only to grade the result.

The first script, `robomme_visual_embed.py`, embeds every frame with SigLIP and writes `emb.npz`. That is one GPU pass per dump.

The second script, `robomme_visual_gate.py`, runs on those embeddings and needs only a CPU. We cut each history wherever the cosine between adjacent frames drops below the threshold. We score each chunk by its highest cosine against the query frame and take the best one. Its frame span is the index we hand over. We also grade that span against the labels and report how pure it is and how much of the lesson it covers.

We use a threshold of 0.923. It falls halfway between the cosine at real session cuts and the lowest cosine inside a session, and we fit it once over the dumps. Holding out any one task family moves it by under 0.001.

The harness that replays these indices through the RoboMME policy server is not here. It produced the file `results/retrieval_rollouts.csv`.
