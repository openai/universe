from universe.rewarder import reward_buffer

def test_prereset():
    buf = reward_buffer.RewardBuffer('buf')
    buf.push('1', 2, False, {'key': 'value'})
    reward, done, info = buf.pop()
    assert reward == 0
    assert done is False
    print(info)

def test_mask_peek():
    buf = reward_buffer.RewardBuffer('buf')
    buf.set_env_info('running', 'test-v0', '1', fps=60)
    buf.push('1', 1, False, {'key': 'value'})
    reward, done, info = buf.pop(peek=True)
    assert info['env_status.episode_id'] is None
    assert info['env_status.env_state'] is None
    assert info['env_status.peek.episode_id'] is None
    assert info['env_status.peek.env_state'] is None

def test_single():
    buf = reward_buffer.RewardBuffer('buf')
    buf.reset('1')
    buf.push('1', 1, False, {'key': 'value'})
    reward, done, info = buf.pop()
    assert reward == 1.0
    assert done is False
    assert info['key'] == 'value'
    assert info['env_status.episode_id'] == '1'
    assert info['env_status.reset.episode_id'] == '1'
    assert info['env.text'] == []

def test_multiple():
    buf = reward_buffer.RewardBuffer('buf')
    buf.reset('1')
    buf.push('1', 1, False, {'key': 'value1'})
    buf.push_text('1', 'text1')

    buf.push('2', 2, False, {'key': 'value2'})
    buf.push_text('2', 'text2')
    buf.push_text('2', 'text3')
    reward, done, info = buf.pop()
    assert reward == 1.0 # old
    assert done is True # old
    assert info['key'] == 'value1', 'Info: {}'.format(info) # old
    assert info['env_status.episode_id'] == '2', 'got: {}, expected: {}'.format(info['env_status.episode_id'], '1')
    assert info['env_status.complete.episode_id'] == '1'
    assert info['env_status.reset.episode_id'] == '1'
    assert info['env.text'] == ['text2', 'text3'] # new

    reward, done, info = buf.pop()
    assert reward == 2.0 # new
    assert done is False
    assert info['key'] == 'value2'
    assert info['env_status.episode_id'] == '2'
    assert 'env_status.reset.episode_id' not in info
    assert info['env.text'] == []

def test_double_reset():
    buf = reward_buffer.RewardBuffer('buf')
    buf.reset('1')
    buf.set_env_info('running', 'test-v0', '1', fps=60)
    buf.push('1', 1, False, {'key': 'value1'})
    buf.set_env_info('resetting', 'test-v0', '2', fps=60)
    buf.push('2', 20, False, {'key': 'value2'})

    reward, done, info = buf.pop(peek=True)
    assert reward == 0
    assert done == False
    assert 'env_status.artificial.done' not in info
    assert info['env_status.episode_id'] == '1'
    assert info['env_status.env_state'] == 'running'
    assert info['env_status.peek.episode_id'] == '2'
    assert info['env_status.peek.env_state'] == 'resetting'

    buf.set_env_info('running', 'test-v0', '2', fps=60)

    reward, done, info = buf.pop(peek=True)
    assert reward == 0
    assert done == False
    assert 'env_status.artificial.done' not in info
    assert info['env_status.episode_id'] == '1'
    assert info['env_status.env_state'] == 'running'
    assert info['env_status.peek.episode_id'] == '2'
    assert info['env_status.peek.env_state'] == 'running'
