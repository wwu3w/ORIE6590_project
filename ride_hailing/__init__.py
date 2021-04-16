from gym.envs.registration import register

register(
    id='RideHailing-v0',
    entry_point='ride_hailing.envs.ride_hailing_env:CityReal',
)
