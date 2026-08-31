// Preview media for the Environments gallery.
//
// Gymnasium environments reuse the official demo GIFs straight from the
// Farama Foundation's own repository (docs/_static/videos) — no need to ship
// or regenerate this media ourselves, and it stays in sync with upstream.
// Board games have no such repository, so they get small local SVG diagrams
// instead (imported as modules so Vite rewrites the URL correctly for the
// packaged file:// build, unlike a hardcoded "/assets/..." path).
import ticTacToePreview from '@/assets/env-previews/tic_tac_toe.svg'
import connectFourPreview from '@/assets/env-previews/connect_four.svg'
import gomokuPreview from '@/assets/env-previews/gomoku.svg'

const GYM_VIDEOS =
  'https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/_static/videos'

function gymGif(folder: string, file: string): string {
  return `${GYM_VIDEOS}/${folder}/${file}.gif`
}

const ENV_PREVIEWS: Record<string, string> = {
  'CartPole-v1': gymGif('classic_control', 'cart_pole'),
  'MountainCar-v0': gymGif('classic_control', 'mountain_car'),
  'MountainCarContinuous-v0': gymGif('classic_control', 'mountain_car_continuous'),
  'Acrobot-v1': gymGif('classic_control', 'acrobot'),
  'Pendulum-v1': gymGif('classic_control', 'pendulum'),

  'FrozenLake-v1': gymGif('toy_text', 'frozen_lake'),
  'Taxi-v4': gymGif('toy_text', 'taxi'),
  'Blackjack-v1': gymGif('toy_text', 'blackjack'),
  'CliffWalking-v1': gymGif('toy_text', 'cliff_walking'),

  'LunarLander-v3': gymGif('box2d', 'lunar_lander'),
  'LunarLanderContinuous-v3': gymGif('box2d', 'lunar_lander'),
  'BipedalWalker-v3': gymGif('box2d', 'bipedal_walker'),
  'BipedalWalkerHardcore-v3': gymGif('box2d', 'bipedal_walker'),
  'CarRacing-v3': gymGif('box2d', 'car_racing'),

  'InvertedPendulum-v5': gymGif('mujoco', 'inverted_pendulum'),
  'InvertedDoublePendulum-v5': gymGif('mujoco', 'inverted_double_pendulum'),
  'Reacher-v5': gymGif('mujoco', 'reacher'),
  'Pusher-v5': gymGif('mujoco', 'pusher'),
  'HalfCheetah-v5': gymGif('mujoco', 'half_cheetah'),
  'Hopper-v5': gymGif('mujoco', 'hopper'),
  'Walker2d-v5': gymGif('mujoco', 'walker2d'),
  'Swimmer-v5': gymGif('mujoco', 'swimmer'),
  'Ant-v5': gymGif('mujoco', 'ant'),
  'Humanoid-v5': gymGif('mujoco', 'humanoid'),
  'HumanoidStandup-v5': gymGif('mujoco', 'humanoid_standup'),

  'ALE/Pong-v5': gymGif('atari', 'pong'),
  'ALE/Breakout-v5': gymGif('atari', 'breakout'),
  'ALE/SpaceInvaders-v5': gymGif('atari', 'space_invaders'),
  'ALE/MsPacman-v5': gymGif('atari', 'ms_pacman'),
  'ALE/Qbert-v5': gymGif('atari', 'qbert'),
  'ALE/Seaquest-v5': gymGif('atari', 'seaquest'),
  'ALE/Enduro-v5': gymGif('atari', 'enduro'),
  'ALE/BeamRider-v5': gymGif('atari', 'beam_rider'),
  'ALE/Asteroids-v5': gymGif('atari', 'asteroids'),
  'ALE/Boxing-v5': gymGif('atari', 'boxing'),
  'ALE/Freeway-v5': gymGif('atari', 'freeway'),
  'ALE/Frostbite-v5': gymGif('atari', 'frostbite'),
  'ALE/Riverraid-v5': gymGif('atari', 'riverraid'),
  'ALE/RoadRunner-v5': gymGif('atari', 'road_runner'),
  'ALE/Assault-v5': gymGif('atari', 'assault'),
  'ALE/Atlantis-v5': gymGif('atari', 'atlantis'),
  'ALE/BattleZone-v5': gymGif('atari', 'battle_zone'),
  'ALE/Centipede-v5': gymGif('atari', 'centipede'),
  'ALE/ChopperCommand-v5': gymGif('atari', 'chopper_command'),
  'ALE/CrazyClimber-v5': gymGif('atari', 'crazy_climber'),
  'ALE/DemonAttack-v5': gymGif('atari', 'demon_attack'),
  'ALE/FishingDerby-v5': gymGif('atari', 'fishing_derby'),
  'ALE/Gopher-v5': gymGif('atari', 'gopher'),
  'ALE/Hero-v5': gymGif('atari', 'hero'),
  'ALE/IceHockey-v5': gymGif('atari', 'ice_hockey'),
  'ALE/Kangaroo-v5': gymGif('atari', 'kangaroo'),
  'ALE/KungFuMaster-v5': gymGif('atari', 'kung_fu_master'),
  'ALE/MontezumaRevenge-v5': gymGif('atari', 'montezuma_revenge'),
  'ALE/Pitfall-v5': gymGif('atari', 'pitfall'),
  'ALE/PrivateEye-v5': gymGif('atari', 'private_eye'),
  'ALE/Skiing-v5': gymGif('atari', 'skiing'),
  'ALE/Tennis-v5': gymGif('atari', 'tennis'),
  'ALE/Asterix-v5': gymGif('atari', 'asterix'),
  'ALE/Phoenix-v5': gymGif('atari', 'phoenix'),
  'ALE/TimePilot-v5': gymGif('atari', 'time_pilot'),
  'ALE/UpNDown-v5': gymGif('atari', 'up_n_down'),
  'ALE/VideoPinball-v5': gymGif('atari', 'video_pinball'),
  'ALE/WizardOfWor-v5': gymGif('atari', 'wizard_of_wor'),
  'ALE/Zaxxon-v5': gymGif('atari', 'zaxxon'),

  tic_tac_toe: ticTacToePreview,
  connect_four: connectFourPreview,
  gomoku: gomokuPreview,
}

export function getEnvPreview(envId: string): string | undefined {
  return ENV_PREVIEWS[envId]
}
