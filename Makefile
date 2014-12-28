all:
	@echo Nothing to do...

ctags:
	ctags --recurse --options=ctags.rust --languages=Rust

docs:
	cargo doc
	# WTF is rustdoc doing?
	in-dir ./target/doc fix-perms
	rscp ./target/doc/* gopher:~/www/burntsushi.net/rustdoc/

clean:
	rm -f $(BUILD)/* $(LIB)
	rm -rf target

push:
	git push origin master
	git push github master

