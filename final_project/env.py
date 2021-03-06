import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer()
        self.viewer.render()


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=800, height=500, resizable=False, caption='Env', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.point = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [50, 50,                # location
                     50, 500,
                     100, 500,
                     100, 50]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.obstacle1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [600, 150,                # location
                     600, 400,
                     610, 400,
                     610, 150]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.obstacle1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [200, 150,              # location
                     200, 160,
                     500, 160,
                     500, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        pass


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
