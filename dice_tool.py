import re
import random

def roll_dice(command):
    """
    Parses a string looking for 'XdY+Z' patterns and rolls them
    """
    # group 1: number of dice (optional, default 1)
    # group 2: number of sides (required)
    # group 3: modifier (optional EX: +5, -2)
    pattern = r"(\d+)?d(\d+)([+-]\d+)?"

    matches = re.findall(pattern, command)

    if not matches:
        return "I didn't see a dice to roll"
    
    results = []
    total_sum = 0

    for match in matches:
        num_str, sides_str, mod_str = match

        num_dice = int(num_str) if num_str else 1
        num_sides = int(sides_str)
        modifier = int(mod_str) if mod_str else 0

        rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
        roll_sum = sum(rolls) + modifier
        total_sum += roll_sum

        roll_str = f"{num_dice}d{num_sides}"
        if modifier != 0:
            roll_str += f"{modifier:+}"

        results.append(f"{roll_str}: {rolls} {'+ ' + str(modifier) if modifier else ''} = **{roll_sum}**")
    return "\n".join(results)

# Test block (only runs if you run this file directly)
if __name__ == "__main__":
    print(roll_dice("I cast fireball for 8d6"))
    print(roll_dice("Roll initiative 1d20+3"))