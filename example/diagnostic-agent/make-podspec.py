#!/usr/bin/env python

def generate_podspec(env_id, cpu_quota, recorder_logdir):
    with open('pod.yaml.template', 'r') as f:
        y = f.read()
        y = y.replace('__ENV_ID__', env_id)
        y = y.replace('__CPU_QUOTA__', cpu_quota)
        y = y.replace('__RECORDER_LOGDIR__', recorder_logdir)
    return y


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('Usage: {} <env_id> <cpu_quota> <recorder_logdir>'.format(sys.argv[0]))
        print('Got: {}'.format(sys.argv))
        sys.exit(1)
    print(generate_podspec(sys.argv[1], sys.argv[2], sys.argv[3]))
