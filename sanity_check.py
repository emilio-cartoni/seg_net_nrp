

def main():

  A_seq = [1] * 5 + [0] * 200
  A_seq = np.random.uniform(0, 1, (50,))
  n_frames, TA = len(A_seq), 1
  n_layers, do_bens_idea = 5, 1
  E_state, R_state = None, None
  for t in range(TA, TA + n_frames):
    A = A_seq[t - TA]
    A_state, E_state, R_state = forward(A, E_state, R_state, n_layers, do_bens_idea)
    plot_output(A_state, E_state, R_state)


def conv(A, weight=0.8):
  # return min(1.0, A * weight)
  return A * weight


def forward(A, E_state, R_state, n_layers, do_bens_idea):

  # Initialization
  A_pile = [0.0 for _ in range(n_layers)]
  E_pile = [0.0 for _ in range(n_layers)]
  R_pile = [0.0 for _ in range(n_layers)]
  if E_state == None:
    E_state = [0.0 for _ in range(n_layers)]
  if R_state == None:
    R_state = [0.0 for _ in range(n_layers)]
  
  # # Bottom-up pass
  # for l in range(n_layers):
  #   A_input = A if l == 0 else E_state[l - 1]
  #   E_pile[l] = abs(conv(A_input) - conv(R_state[l]))
  #   A_pile[l] = conv(A_input)  # bu_drop and bu_conv (just here to plot)

  # # Top-down pass
  # for l in reversed(range(n_layers)):
  #   if l == n_layers - 1:
  #     R_input = E_state[l]
  #   else:
  #     if do_bens_idea:
  #       R_input = E_state[l] * R_state[l + 1]  # ommited td_upsp on R_state
  #     else:
  #       R_input = E_state[l] + R_state[l + 1]  # ommited td_upsp on R_state, concat
  #   R_pile[l] = conv(R_input + R_state[l])  # td_conv (hgru)

  # Top-down pass
  for l in reversed(range(n_layers)):
    R = R_state[l]
    E = E_state[l]
    if l != n_layers - 1:
      if do_bens_idea:
        E = E * (R_state[l + 1])
      else:
        E = E + R_state[l + 1]
    R_pile[l] = conv(E + R)

  # Bottom-up pass
  for l in range(n_layers):
    A = conv(A)
    A_hat = conv(R_state[l])
    E_pile[l] = abs(A - A_hat)
    A_pile[l] = A
    if l < n_layers - 1:
      A = E_state[l]

  # Update network and send output
  print(
    [f'{A:.1f}' for A in A_pile],
    [f'{E:.1f}' for E in E_pile],
    [f'{R:.1f}' for R in R_pile]
  )
  return A_pile, E_pile, R_pile


import numpy as np
import matplotlib.pyplot as plt
def plot_output(A, E, R):

  Z = [0.0] * len(A)
  A = np.expand_dims(np.insert(Z, np.arange(len(A)), A)[:-1], axis=-1)
  E = np.expand_dims(np.insert(Z, np.arange(len(E)), E)[:-1], axis=-1)
  R = np.expand_dims(np.insert(Z, np.arange(len(R)), R)[:-1], axis=-1)
  Z = np.expand_dims(np.insert(Z, np.arange(len(Z)), Z)[:-1], axis=-1)
  plot = np.flipud(np.concatenate((A, E, R), axis=1))
  print(plot.shape)
  plt.imshow(plot, cmap='Greys', vmin=0, vmax=1)
  plt.show()


if __name__ == '__main__':
  main()