clear
cat << EOF
pruning docker by removing...   
      - all stopped containers               
      - all networks not used by at least one container                        
      - all dangling images                  
      - all build cache                      

                        ##         .
                 ## ## ##        ==
              ## ## ## ## ##    ===
          /""""""""""""""""""\___/ ===
    ~~~ {~~ ~~~~ ~~~ ~~~~ ~~~ ~ /  ===- ~~~
          \______ o           __/
            \    \         __/
             \____\_______/
EOF

docker system prune -f
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose build
docker-compose up
docker-compose down