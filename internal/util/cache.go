package util

import (
	"context"
	"github.com/go-redis/redis/v8"
	"github.com/spf13/viper"
	"log"
)

func RedisFlush() (string, error) {
	host := viper.GetString("redis.host")
	port := viper.GetString("redis.port")
	addr := host + ":" + port

	rdb := redis.NewClient(&redis.Options{
		Addr: addr,
	})
	defer rdb.Close()

	ctx := context.Background()
	defer ctx.Done()
	if val, err := rdb.FlushAll(ctx).Result(); err != nil {
		return val, err
	} else {
		log.Printf("redis flushall: %v\n", val)
		return val, nil
	}
}

func RedisSet(key, value string) error {
	host := viper.GetString("redis.host")
	port := viper.GetString("redis.port")
	addr := host + ":" + port

	rdb := redis.NewClient(&redis.Options{
		Addr: addr,
	})
	defer rdb.Close()

	ctx := context.Background()
	defer ctx.Done()
	if _, err := rdb.Set(ctx, key, value, 0).Result(); err != nil {
		return err
	} else {
		//log.Printf("redis: set={%v}\n", val)
		return nil
	}
}
