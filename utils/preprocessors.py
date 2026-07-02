def keep_last_k_messages(k):
    def _fn(state):
        state['messages'] = state['messages'][-k:] if k > 0 else []
        return state
    return _fn

clear_messages_from_state = keep_last_k_messages(0)
keep_last_1_message = keep_last_k_messages(1)

preprocessors_dict = {
    'CLEAR_ALL_MESSAGES': clear_messages_from_state,
    'KEEP_ONLY_LAST_MESSAGE': keep_last_1_message,
}