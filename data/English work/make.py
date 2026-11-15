from moviepy.editor import *
from moviepy.video.tools.drawing import color_gradient
from moviepy.editor import TextClip, ColorClip, CompositeVideoClip, AudioFileClip
from moviepy.editor import ImageClip
import numpy as np


# 创建一个函数来生成每个场景的内容
def create_scene(text, subtitle, start_time, duration, background_color=(255, 255, 255), character_image=None):
    # 创建背景色的片段（例如：一个白色背景）
    background = ColorClip(size=(1920, 1080), color=background_color, duration=duration)

    # 创建英文台词文本
    text_clip_en = TextClip(text, fontsize=50, color='black', font='Amiri', size=(1920, 1080))
    text_clip_en = text_clip_en.set_position('center').set_duration(duration).set_start(start_time)

    # 创建中文字幕
    text_clip_cn = TextClip(subtitle, fontsize=50, color='white', font='Amiri', size=(1920, 1080))
    text_clip_cn = text_clip_cn.set_position(('center', 600)).set_duration(duration).set_start(start_time)

    # 如果有虚拟人物图像，添加它
    if character_image:
        character_clip = ImageClip(character_image).resize(height=500).set_position(('center', 'center')).set_duration(
            duration).set_start(start_time)
        # 组合视频片段
        scene = CompositeVideoClip([background, character_clip, text_clip_en, text_clip_cn])
    else:
        # 如果没有虚拟人物图像，直接组合背景和字幕
        scene = CompositeVideoClip([background, text_clip_en, text_clip_cn])

    scene = scene.set_duration(duration).set_start(start_time)
    return scene


# 添加特殊的视觉元素效果
def add_special_effects(clip, effect_type):
    if effect_type == "slow_motion":
        # 添加慢动作效果
        return clip.fx(vfx.speedx, 0.5)
    elif effect_type == "gradient":
        # 添加渐变效果
        gradient_clip = ColorClip(size=(1920, 1080), color=(255, 0, 0), duration=clip.duration)
        return CompositeVideoClip([clip, gradient_clip])
    elif effect_type == "starry_sky":
        # 生成星空效果（假设我们有一个星空的图像）
        starry_sky = ImageClip("starry_sky_image.jpg").resize((1920, 1080)).set_duration(clip.duration)
        return CompositeVideoClip([clip, starry_sky])
    elif effect_type == "waveform":
        # 生成一个心电图样的效果（假设我们有图像）
        waveform = ImageClip("waveform_image.jpg").resize((1920, 1080)).set_duration(clip.duration)
        return CompositeVideoClip([clip, waveform])


# 创建每个时间段的场景
scene1 = create_scene("What you're seeing isn't failure... it's the birthplace of a scientist's conscience.",
                      "你眼前不是失败…而是一个科学家良知的诞生地", 0, 22, character_image="virtual_character1.png")
scene1 = add_special_effects(scene1, "slow_motion")  # 慢镜头效果

scene2 = create_scene(
    "Day 129: Still can't replicate the results. My advisor says 'Try harder'... but what if the problem isn't in the data, but in the question itself?",
    "第129天：仍无法复现结果。导师说『再加把劲』…但如果问题不在数据，而在问题本身呢？", 23, 45,
    character_image="virtual_character2.png")
scene2 = add_special_effects(scene2, "gradient")  # 渐变效果

scene3 = create_scene(
    "The explosion sent me here. While waiting for skin grafts, I met 8-year-old Mia - her body failing, but her mind firing endless 'whys' about my bandages.",
    "爆炸把我送来这里。等待植皮时，我遇见8岁的米娅——她的身体在衰竭，却对我的绷带发出无数『为什么』", 46, 90,
    character_image="virtual_character3.png")
scene3 = add_special_effects(scene3, "starry_sky")  # 星空效果

scene4 = create_scene(
    "Her questions became my new lab notebook. That's when I saw it: We chase 'significant p-values', but forget science's first purpose - to ease human wonder and suffering.",
    "她的问题成了我的新实验笔记。那一刻我明白了：我们追逐『显著性p值』，却忘了科学的初心——安抚人类的好奇与苦难", 91, 135,
    character_image="virtual_character4.png")
scene4 = add_special_effects(scene4, "waveform")  # 心电波形效果

scene5 = create_scene(
    "I rebuilt the lab with wheelchair access. Now my 'failed' compound helps kids like Mia watch experiments from their hospital beds via AR goggles.",
    "我把实验室改造成轮椅可进。现在那个『失败』的化合物，正通过AR眼镜让米娅这样的孩子在病床上参与实验", 136, 160,
    character_image="virtual_character5.png")
scene5 = add_special_effects(scene5, "gradient")  # 渐变效果

scene6 = create_scene(
    "They call it 'translational research'... but true translation isn't from lab to clinic, it's from a scientist's ego to a child's eyes.",
    "人们称之为『转化医学』…但真正的转化不是从实验室到病房，而是从科学家的自负到孩童的眼睛", 161, 180,
    character_image="virtual_character6.png")
scene6 = add_special_effects(scene6, "starry_sky")  # 星空效果

# 合并所有场景
final_video = concatenate_videoclips([scene1, scene2, scene3, scene4, scene5, scene6])

# 添加背景音乐（假设我们有一个音频文件）
audio = AudioFileClip("background_music.mp3")
final_video = final_video.set_audio(audio)

# 导出最终的视频
final_video.write_videofile("final_video_with_virtual_character.mp4", fps=24)
