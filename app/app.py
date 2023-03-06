import base64
import json
import requests
import time
import onnxruntime as ort
from rembg.session_simple import SimpleSession
from rembg import remove

# init code
print("initializing model...")
start = time.time_ns()/1e6
sess_opts = ort.SessionOptions()
sess_opts.inter_op_num_threads = 1
session = SimpleSession("u2net", ort.InferenceSession(
    str('/var/task/.u2net/u2net.onnx'),
    providers=ort.get_available_providers(),
    sess_options=sess_opts,
),
)
print(f"model initialized: {(time.time_ns()/1e6 - start):.1f}ms")


def remove_bg(base64_image):
    photo_bytes = base64.b64decode(base64_image.split(',')[-1])
    print(photo_bytes)
    image = Image.open(io.BytesIO(photo_bytes)).convert('RGB')
    return remove(input)


def handler(event, context):
    print(json.dumps(event))
    base64_image = ''
    if 'base64_image' in event:
        base64_image = event['base64_image']
    if base64_image == '':
        return {
            'headers': {"Content-Type": "application/json"},
            'statusCode': 400,
            'body': json.dumps({'error': 'bad request'})
        }

    try:
        photo_bytes = base64.b64decode(base64_image.split(',')[-1])
        print(photo_bytes)
        # too slow: , alpha_matting=True)
        output = remove(photo_bytes, session=session)
        return {
            'headers': {"Content-Type": "image/png"},
            'statusCode': 200,
            'body': base64.b64encode(output).decode('utf-8'),
            'isBase64Encoded': True
        }
    except Exception as ex:
        return {
            'headers': {"Content-Type": "application/json"},
            'statusCode': 500,
            'body': json.dumps({'error': ex.__repr__()})
        }