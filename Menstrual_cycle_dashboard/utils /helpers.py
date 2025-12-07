def decode_phase(phase_encoded):
    """Decode numeric phase encoding to human-readable string"""
    phase_map = {
        1: 'Follicular',
        2: 'Fertility',
        3: 'Luteal',
        4: 'Menstrual'
    }
    return phase_map.get(phase_encoded, 'Unknown')
