from sshkeyboard import listen_keyboard
from MBot.Messages.message_defs import mo_states_dtype, mo_cmds_dtype, mo_pid_params_dtype
from MBot.SerialProtocol.protocol import SerialProtocol

def press(key):
    if key == "up":
        print("up pressed")
    elif key == "down":
        print("down pressed")
    elif key == "left":
        print("left pressed")
    elif key == "right":
        print("right pressed")

listen_keyboard(on_press=press)