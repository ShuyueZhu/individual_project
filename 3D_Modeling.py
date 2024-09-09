import bpy
import os
import math

# 设置图片文件夹路径和渲染输出路径
image_folder = "D:\\ic\\binary\\healthy"
output_folder = "D:\\ic\\rotate\\healthy"

# 设置Twist修改器的角度（以弧度表示）
twist_angle = math.radians(20)  # 20度的扭曲，转换为弧度

# 设置相机旋转角度（直接使用弧度）
camera_rotation = (math.radians(60), math.radians(30), math.radians(45))  # 60°, 0°, 45° 转换为弧度

# 设置背景颜色为黑色
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0, 0, 0, 1)  # RGB为(0, 0, 0)，即黑色，Alpha为1

# 设置渲染输出的分辨率为384×384
bpy.context.scene.render.resolution_x = 384
bpy.context.scene.render.resolution_y = 384
bpy.context.scene.render.resolution_percentage = 100  # 确保输出100%分辨率

# 设置图像质量为JPG（RGB格式）
bpy.context.scene.render.image_settings.file_format = 'JPEG'  # 设置保存为JPEG格式
bpy.context.scene.render.image_settings.quality = 100  # 设置JPEG质量为100，保持最高质量
bpy.context.scene.render.image_settings.color_mode = 'RGB'  # 确保不包含Alpha通道

# 删除所有现有光源
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.data.objects:
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)

if not any(obj.type == 'LIGHT' and obj.data.type == 'SUN' for obj in bpy.data.objects):
    sun_light_data = bpy.data.lights.new(name="Sun Light", type='SUN')
    sun_light_data.energy = 5.0  # 设置太阳光强度
    sun_light_object = bpy.data.objects.new(name="Sun Light", object_data=sun_light_data)
    bpy.context.collection.objects.link(sun_light_object)

    # 设置太阳光的位置和方向（使用弧度值）
    sun_light_object.location = (10, 10, 10)
    sun_light_object.rotation_euler = (math.radians(60), math.radians(-30), math.radians(-45))

# 批量处理图片
for image_name in os.listdir(image_folder):
    if image_name.endswith((".png", ".jpg", ".jpeg")):
        # 1. 添加图片作为平面
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.mesh.primitive_plane_add()
        bpy.ops.image.open(filepath=os.path.join(image_folder, image_name))
        img = bpy.data.images.load(os.path.join(image_folder, image_name))
        mat = bpy.data.materials.new(name="ImageMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = img

        # 2. 使用发射材质，使图片不受光照影响
        emission_shader = mat.node_tree.nodes.new('ShaderNodeEmission')
        mat.node_tree.links.new(emission_shader.inputs['Color'], tex_image.outputs['Color'])
        mat.node_tree.links.new(mat.node_tree.nodes['Material Output'].inputs['Surface'],
                                emission_shader.outputs['Emission'])

        obj = bpy.context.object
        obj.data.materials.append(mat)

        #        # 1. 添加图片作为平面
        #        bpy.ops.object.select_all(action='DESELECT')
        #        bpy.ops.mesh.primitive_plane_add()
        #        bpy.ops.image.open(filepath=os.path.join(image_folder, image_name))
        #        img = bpy.data.images.load(os.path.join(image_folder, image_name))
        #        mat = bpy.data.materials.new(name="ImageMaterial")
        #        mat.use_nodes = True
        #        bsdf = mat.node_tree.nodes["Principled BSDF"]
        #        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        #        tex_image.image = img
        #        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
        #        obj = bpy.context.object
        #        obj.data.materials.append(mat)

        # 3. 细分平面网格，使其更加平滑
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=10)  # 细分10次

        # 4. 添加Subsurf细分曲面修改器，设置细分等级
        bpy.ops.object.mode_set(mode='OBJECT')
        subsurf_modifier = obj.modifiers.new(name="Subsurf", type='SUBSURF')
        subsurf_modifier.levels = 2  # 设置预览细分等级为2
        subsurf_modifier.render_levels = 2  # 设置渲染时的细分等级为2

        # 5. 添加Twist修改器，并使用弧度值
        twist_modifier = obj.modifiers.new(name="Twist", type='SIMPLE_DEFORM')
        twist_modifier.deform_method = 'TWIST'
        twist_modifier.deform_axis = 'X'
        twist_modifier.angle = twist_angle  # 直接使用弧度值

        # 6. 设置相机视角，使用弧度值
        bpy.context.scene.camera.location = (0, -3, 3)
        bpy.context.scene.camera.rotation_euler = camera_rotation

        #        # 添加自然太阳光
        #        sun_light_data = bpy.data.lights.new(name="Sun Light", type='SUN')
        #        sun_light_data.energy = 5.0  # 设置太阳光强度
        #        sun_light_object = bpy.data.objects.new(name="Sun Light", object_data=sun_light_data)
        #        bpy.context.collection.objects.link(sun_light_object)
        #
        #        # 设置太阳光的位置和方向（使用弧度值）
        #        sun_light_object.location = (10, 10, 10)
        #        sun_light_object.rotation_euler = (math.radians(30), math.radians(-30), math.radians(-45))  # 使用弧度值设置方向

        # 7. 将当前视角设为相机视角
        bpy.context.view_layer.objects.active = obj
        bpy.ops.view3d.camera_to_view_selected()

        # 8. 渲染并输出图像
        bpy.context.scene.render.filepath = os.path.join(output_folder, image_name.split('.')[0] + '.jpg')
        bpy.context.scene.render.image_settings.compression = 0  # 设置压缩等级为0（无损压缩）
        bpy.ops.render.render(write_still=True)

        # 9. 删除处理完的对象
        bpy.ops.object.delete()

# 恢复相机默认位置（可选）
bpy.context.scene.camera.location = (0, -10, 10)
bpy.context.scene.camera.rotation_euler = (1.109319, 0.0, 0.785398)  # 使用弧度设置默认角度
