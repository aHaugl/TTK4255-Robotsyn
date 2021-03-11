axis_points = [tf.new_points(f'{tf}_ax', 0.05*np.eye(3)) for tf in tfs]
axis_points_w_origos = [[tf.origo, point] for tf, axis in zip(tfs, axis_points)
                        for point in axis]


axis_points = camera.project_points(flatten(axis_points_w_origos))
axis_projections = project(K, axis_points)
f = sp.lambdify(rotors.free_symbols, axis_projections)

fig, ax = plt.subplots(1, 1)

compensation = np.array([11.6, 28.9, 0.0])*np.pi/180 - logs[0, 1:]
rotations = logs[img_idx*16, 1:] + compensation
lines = f(*rotations)
x, y = lines
plt.imshow(img)

for i in range(0, lines.shape[1], 2):
    ax.plot(*lines[:, i:i+2], color='rgb'[(i//2) % 3], linewidth=2)
