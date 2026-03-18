import os

_HEADLESS = os.environ.get("SDL_VIDEODRIVER") == "dummy"

dir_path = os.path.dirname(os.path.realpath(__file__))
resource_path = os.path.join(dir_path, 'resources')

if not _HEADLESS:
    try:
        import pyglet
        from . import util

        pyglet.resource.path = [resource_path]
        pyglet.resource.reindex()

        sprite_map = pyglet.resource.image('spritemap.png')

        s0 = sprite_map.get_region(3, 9, 17, 12)
        s1 = sprite_map.get_region(31, 9, 17, 12)
        s2 = sprite_map.get_region(59, 9, 17, 12)
        player_anim = pyglet.image.Animation.from_image_sequence([s0, s1, s2], 0.1, True)
        player_image = s0

        background_image = sprite_map.get_region(0, 512-256, 144, 256)

        wall1_image = sprite_map.get_region(56, 512-483, 26, 160)
        wall2_image = sprite_map.get_region(84, 512-483, 26, 160)

        ground_image = sprite_map.get_region(292, 512-55, 168, 56)

        util.center_image(s0)
        util.center_image(s1)
        util.center_image(s2)
        util.center_image(wall1_image)
        util.center_image(wall2_image)

        hurt_file = os.path.join(dir_path, 'resources/hurt.wav')
        flap_file = os.path.join(dir_path, 'resources/flap.wav')
        score_file = os.path.join(dir_path, 'resources/score.wav')

        hurt_sound = pyglet.media.load(hurt_file, streaming=False)
        flap_sound = pyglet.media.load(flap_file, streaming=False)
        score_sound = pyglet.media.load(score_file, streaming=False)
    except Exception:
        _HEADLESS = True

if _HEADLESS:
    # Stub objects for headless mode
    player_anim = None
    player_image = None
    background_image = None
    wall1_image = None
    wall2_image = None
    ground_image = None
    hurt_sound = None
    flap_sound = None
    score_sound = None

    def center_image(img):
        pass
