// Keyed by the text each printed QR code decodes to.
const MICE = {
  'laptop': {
    name: 'Jerry',
    color: '#f28fad',
    tags: ['Calm', 'Gentle', 'Independent', 'Cozy', 'Sweet'],
  },
  'headphones': {
    name: 'Marshmallow',
    color: '#8ec5ff',
    tags: ['Playful', 'Energetic', 'Social', 'Cozy', 'Curious', 'Bold', 'Charming'],
  },
  'wallet': {
    name: 'Tequila',
    color: '#ffb347',
    tags: ['Energetic', 'Independent', 'Free-spirited', 'Curious', 'Dreamy'],
  },
  'keys': {
    name: 'Sam',
    color: '#b58cff',
    tags: ['Playful', 'Social', 'Free-spirited', 'Curious', 'Bold'],
  },
  'cup': {
    name: 'Ham',
    color: '#5fd4b8',
    tags: ['Gentle', 'Social', 'Cozy', 'Sweet', 'Sensitive', 'Charming'],
  },
  'bottle-full': {
    name: 'GusGus',
    color: '#9bd36a',
    tags: ['Calm', 'Gentle', 'Independent', 'Curious', 'Sweet', 'Dreamy', 'Sensitive'],
  },
  'bottle-half': {
    name: 'Shrek',
    color: '#ffd166',
    tags: ['Playful', 'Energetic', 'Free-spirited', 'Bold'],
  },
  'bottle-empty': {
    name: 'Taro',
    color: '#c9a0dc',
    tags: ['Calm', 'Independent', 'Dreamy', 'Sensitive', 'Charming'],
  },
};

// Hand-written personalities for specific groups of mice. The key is the QR
// ids of the group, sorted alphabetically and joined with '+'.
const GROUP_PERSONALITIES = {
  'cup+keys': {
    title: 'Ham & Sam: The Inseparables',
    tags: ['Loyal', 'Mischievous', 'Finish each other\'s squeaks'],
  },
  'bottle-half+headphones': {
    title: 'Chaos Crew',
    tags: ['Unstoppable', 'Loud', 'Midnight zoomies'],
  },
  'bottle-empty+laptop': {
    title: 'Quiet Philosophers',
    tags: ['Thoughtful', 'Serene', 'Nap enthusiasts'],
  },
};

// Any other group gets a personality from the "chemistry" between its
// members: when one mouse has the first trait and a different mouse has the
// second, together they spark the new trait.
const CHEMISTRY = [
  ['Calm', 'Energetic', 'Balanced'],
  ['Playful', 'Social', 'Party Animals'],
  ['Bold', 'Sensitive', 'Protective'],
  ['Curious', 'Dreamy', 'Adventurous'],
  ['Independent', 'Social', 'Push & Pull'],
  ['Cozy', 'Sweet', 'Cuddle Buddies'],
  ['Calm', 'Free-spirited', 'Easygoing'],
  ['Bold', 'Charming', 'Show-offs'],
  ['Gentle', 'Energetic', 'Patient'],
  ['Independent', 'Independent', 'Respectful'],
];

// hasOwn guards against QR text like "constructor" resolving to an inherited
// Object property instead of a mouse.
function getMouse(id) {
  return Object.hasOwn(MICE, id) ? { id, ...MICE[id] } : null;
}

function getGroupPersonality(mice) {
  const key = mice.map((mouse) => mouse.id).sort().join('+');
  if (Object.hasOwn(GROUP_PERSONALITIES, key)) {
    return GROUP_PERSONALITIES[key];
  }

  const sparked = [];
  for (const [a, b, result] of CHEMISTRY) {
    const sparks = mice.some((first) => first.tags.includes(a)
      && mice.some((second) => second !== first && second.tags.includes(b)));
    if (sparks) {
      sparked.push(result);
    }
  }

  const shared = mice[0].tags.filter((tag) => mice.every((mouse) => mouse.tags.includes(tag)));
  const together = mice.length === 2 ? 'Both' : 'All';
  const tags = [...sparked, ...shared.map((tag) => `${together} ${tag}`)].slice(0, 5);

  return {
    title: mice.map((mouse) => mouse.name).join(' & '),
    tags: tags.length > 0 ? tags : ['Still getting to know each other'],
  };
}
