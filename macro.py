
def run(y_center, z_center, path, dry_run=False, step_size=(0.1, 0.1)):

    # y to the right, z downwards
    points = [
        'ABECD',
        'BAFDC',
        'ABXCD',
        'BAGDC',
        'ABHCD',
    ]

    offsets = (2, 2)  # Center point should have coordinates [0, 0]

    for name in path:
        for z, row in enumerate(points):
            for y, point in enumerate(row):
                if point == name:
                    y_real = y_center + step_size[0] * (y - offsets[0])
                    z_real = z_center + step_size[1] * (z - offsets[1])

                    if dry_run:
                        print(y_real, z_real)
                    else:
                        mv(sy, y_real)
                        mv(sz, z_real)
                        NiKa_xanes2.scan()


if __name__ == '__main__':
    run(y_center=3.0, z_center=2.0, path='AC', dry_run=True)
